import os
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from core import utils
from optim.LS import LabelSmoothingCrossEntropy
from pruner import (
    check_sparsity,
    extract_mask,
    global_prune_model,
    prune_model_custom,
    pruning_model,
    pruning_model_random,
    remove_prune,
)
from trainer import train as default_train
from trainer import train_sam_epoch
from trainer import validate


def _imagenet_lambda_scheduler(args, optimizer, total_epochs):
    lambda0 = (
        lambda cur_iter: (cur_iter + 1) / args.warmup
        if cur_iter < args.warmup
        else (
            0.5 * (1.0 + np.cos(np.pi * ((cur_iter - args.warmup) / (total_epochs - args.warmup))))
        )
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)


def _build_scheduler(args, optimizer, *, mode):
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    if args.imagenet_arch:
        return _imagenet_lambda_scheduler(args, optimizer, args.epochs)
    if mode == "vit_initial":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )


def _build_optimizer(args, model, *, mode):
    if mode == "sam_initial":
        from optim.SAM import SAM

        return SAM(
            model.parameters(),
            torch.optim.SGD,
            rho=2.0,
            adaptive=True,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if mode == "vit_initial":
        return torch.optim.Adam(model.parameters(), args.lr)
    return torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def _get_profile(profile_name):
    if profile_name == "imp":
        return {
            "criterion": torch.nn.CrossEntropyLoss(),
            "train_fn": default_train,
            "initial_optimizer_mode": "sgd",
            "initial_scheduler_mode": "multistep",
        }
    if profile_name == "ls":
        return {
            "criterion": LabelSmoothingCrossEntropy(),
            "train_fn": default_train,
            "initial_optimizer_mode": "sgd",
            "initial_scheduler_mode": "multistep",
        }
    if profile_name == "sam":
        return {
            "criterion": torch.nn.CrossEntropyLoss(),
            "train_fn": train_sam_epoch,
            "initial_optimizer_mode": "sam_initial",
            "initial_scheduler_mode": "multistep",
        }
    if profile_name == "vit":
        return {
            "criterion": torch.nn.CrossEntropyLoss(),
            "train_fn": default_train,
            "initial_optimizer_mode": "vit_initial",
            "initial_scheduler_mode": "vit_initial",
        }
    raise ValueError(f"Unknown profile: {profile_name}")


def run_pruning(args, profile_name):
    profile = _get_profile(profile_name)
    print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)

    if args.dataset == "imagenet":
        args.class_to_replace = None
        model, train_loader, val_loader = utils.setup_model_dataset(args)
    else:
        model, train_loader, val_loader, _, _ = utils.setup_model_dataset(args)
    model.cuda()

    criterion = profile["criterion"]
    train_fn = profile["train_fn"]

    if args.prune_type == "lt":
        print("lottery tickets setting (rewind to the same random init)")
        initialization = deepcopy(model.state_dict())
    elif args.prune_type == "pt":
        print("lottery tickets from best dense weight")
        initialization = None
    elif args.prune_type == "rewind_lt":
        print("lottery tickets with early weight rewinding")
        initialization = None
    else:
        raise ValueError("unknown prune_type")

    optimizer = _build_optimizer(args, model, mode=profile["initial_optimizer_mode"])
    scheduler = _build_scheduler(args, optimizer, mode=profile["initial_scheduler_mode"])

    best_sa = 0
    if args.resume:
        print("resume from checkpoint {}".format(args.checkpoint))
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device("cuda:" + str(args.gpu))
        )
        best_sa = checkpoint["best_sa"]
        print(best_sa)
        start_epoch = checkpoint["epoch"]
        all_result = checkpoint["result"]
        start_state = checkpoint["state"]
        print(start_state)
        if start_state > 0:
            current_mask = extract_mask(checkpoint["state_dict"])
            prune_model_custom(model, current_mask)
            check_sparsity(model)
            optimizer = _build_optimizer(args, model, mode="sgd")
            scheduler = _build_scheduler(args, optimizer, mode="multistep")

        model.load_state_dict(checkpoint["state_dict"], strict=False)
        x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
        model.eval()
        with torch.no_grad():
            model(x_rand)

        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        initialization = checkpoint["init_weight"]
        print("loading state:", start_state)
        print("loading from epoch: ", start_epoch, "best_sa=", best_sa)
    else:
        all_result = {"train_ta": [], "test_ta": [], "val_ta": []}
        start_epoch = 0
        start_state = 0

    print(
        "######################################## Start Standard Training Iterative Pruning ########################################"
    )

    total_training_rounds = args.pruning_times + 1
    for state in range(start_state, total_training_rounds):
        print("******************************************")
        print("training/pruning state", state)
        print("******************************************")

        check_sparsity(model)
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            print(optimizer.state_dict()["param_groups"][0]["lr"])
            acc = train_fn(train_loader, model, criterion, optimizer, epoch, args)

            if state == 0 and (epoch + 1) == args.rewind_epoch:
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_dir, f"epoch_{epoch + 1}_rewind_weight.pt"),
                )
                if args.prune_type == "rewind_lt":
                    initialization = deepcopy(model.state_dict())

            tacc = validate(val_loader, model, criterion, args)
            scheduler.step()

            all_result["train_ta"].append(acc)
            all_result["val_ta"].append(tacc)

            is_best_sa = tacc > best_sa
            best_sa = max(tacc, best_sa)

            utils.save_checkpoint(
                {
                    "state": state,
                    "result": all_result,
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_sa": best_sa,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "init_weight": initialization,
                },
                is_SA_best=is_best_sa,
                pruning=state,
                save_path=args.save_dir,
            )

            plt.plot(all_result["train_ta"], label="train_acc")
            plt.plot(all_result["val_ta"], label="val_acc")
            plt.plot(all_result["test_ta"], label="test_acc")
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(state) + "net_train.png"))
            plt.close()
            print("one epoch duration:{}".format(time.time() - start_time))

        check_sparsity(model)
        print("Performance on the test data set")
        validate(val_loader, model, criterion, args)
        if len(all_result["val_ta"]) != 0:
            val_pick_best_epoch = np.argmax(np.array(all_result["val_ta"]))
            print(
                "* best SA = {}, Epoch = {}".format(
                    all_result["val_ta"][val_pick_best_epoch], val_pick_best_epoch + 1
                )
            )

        all_result = {"train_ta": [], "test_ta": [], "val_ta": []}
        best_sa = 0
        start_epoch = 0

        # With the new semantics, pruning_times means "number of pruning steps".
        # We therefore run one extra final training round without another prune.
        if state == args.pruning_times:
            print("final training round reached; skip pruning and finish")
            continue

        if args.prune_type == "pt":
            print("* loading pretrained weight")
            initialization = torch.load(
                os.path.join(args.save_dir, "0model_SA_best.pth.tar"),
                map_location=torch.device("cuda:" + str(args.gpu)),
            )["state_dict"]

        if args.random_prune:
            print("random pruning")
            pruning_model_random(model, args.rate)
        else:
            print("L1 pruning")
            pruning_model(model, args.rate)

        current_mask = extract_mask(model.state_dict())
        remove_prune(model)

        model.load_state_dict(initialization, strict=False)
        prune_model_custom(model, current_mask)
        optimizer = _build_optimizer(args, model, mode="sgd")
        scheduler = _build_scheduler(args, optimizer, mode="multistep")
        if args.rewind_epoch:
            for _ in range(args.rewind_epoch):
                scheduler.step()


def run_synflow(args):
    print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)

    setup = utils.setup_model_dataset(args)
    if len(setup) == 5:
        model, train_loader, val_loader, test_loader, _ = setup
    elif len(setup) == 4:
        model, train_loader, val_loader, test_loader = setup
    elif len(setup) == 3:
        model, train_loader, val_loader = setup
        test_loader = val_loader
    else:
        raise ValueError("Unexpected dataset setup output for SynFlow.")
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    if args.prune_type == "lt":
        print("lottery tickets setting (rewind to the same random init)")
        initialization = deepcopy(model.state_dict())
    elif args.prune_type == "pt":
        print("lottery tickets from best dense weight")
        initialization = None
    elif args.prune_type == "rewind_lt":
        print("lottery tickets with early weight rewinding")
        initialization = None
    else:
        raise ValueError("unknown prune_type")

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )

    best_sa = 0
    if args.resume:
        print("resume from checkpoint {}".format(args.checkpoint))
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device("cuda:" + str(args.gpu))
        )
        best_sa = checkpoint["best_sa"]
        start_epoch = checkpoint["epoch"]
        all_result = checkpoint["result"]
        start_state = checkpoint["state"]

        if start_state > 0:
            current_mask = extract_mask(checkpoint["state_dict"])
            prune_model_custom(model, current_mask)
            check_sparsity(model)
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )

        model.load_state_dict(checkpoint["state_dict"], strict=False)
        x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
        model.eval()
        with torch.no_grad():
            model(x_rand)

        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        initialization = checkpoint["init_weight"]
        print("loading state:", start_state)
        print("loading from epoch: ", start_epoch, "best_sa=", best_sa)
        check_sparsity(model)
    else:
        all_result = {"train_ta": [], "test_ta": [], "val_ta": []}
        start_epoch = 0

    print(
        "######################################## Start Standard Synflow Pruning ########################################"
    )

    if args.rate != 0:
        global_prune_model(model, args.rate, "synflow", train_loader)
        check_sparsity(model)

    check_sparsity(model)
    state = 0
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(optimizer.state_dict()["param_groups"][0]["lr"])
        acc = default_train(train_loader, model, criterion, optimizer, epoch, args)

        tacc = validate(val_loader, model, criterion, args)
        test_tacc = validate(test_loader, model, criterion, args)

        scheduler.step()

        all_result["train_ta"].append(acc)
        all_result["val_ta"].append(tacc)
        all_result["test_ta"].append(test_tacc)

        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        utils.save_checkpoint(
            {
                "state": state,
                "result": all_result,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_sa": best_sa,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "init_weight": initialization,
            },
            is_SA_best=is_best_sa,
            pruning=state,
            save_path=args.save_dir,
        )

        plt.plot(all_result["train_ta"], label="train_acc")
        plt.plot(all_result["val_ta"], label="val_acc")
        plt.plot(all_result["test_ta"], label="test_acc")
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, str(state) + "net_train.png"))
        plt.close()
        print("one epoch duration:{}".format(time.time() - start_time))

    check_sparsity(model)
    print("Performance on the test data set")
    validate(test_loader, model, criterion, args)
    if len(all_result["val_ta"]) != 0:
        val_pick_best_epoch = np.argmax(np.array(all_result["val_ta"]))
        print(
            "* best SA = {}, Epoch = {}".format(
                all_result["test_ta"][val_pick_best_epoch], val_pick_best_epoch + 1
            )
        )
