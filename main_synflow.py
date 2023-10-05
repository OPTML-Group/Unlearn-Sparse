"""
    main process for a Lottery Tickets experiments
"""
import argparse
import os
import pdb
import pickle
import random
import shutil
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import arg_parser
from pruner import *
from trainer import train, validate
from utils import *
from utils import NormalizeByChannelMeanStd

best_sa = 0


def main():
    global args, best_sa
    args = arg_parser.parse_args()
    print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        model, train_loader, val_loader, test_loader, _ = setup_model_dataset(args)
    else:
        model, train_loader, val_loader, test_loader = setup_model_dataset(args)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

    if args.prune_type == "lt":
        print("lottery tickets setting (rewind to the same random init)")
        initalization = deepcopy(model.state_dict())
    elif args.prune_type == "pt":
        print("lottery tickets from best dense weight")
        initalization = None
    elif args.prune_type == "rewind_lt":
        print("lottery tickets with early weight rewinding")
        initalization = None
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
    )  # 0.1 is fixed

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
        # adding an extra forward process to enable the masks
        x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
        model.eval()
        with torch.no_grad():
            model(x_rand)

        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        initalization = checkpoint["init_weight"]
        print("loading state:", start_state)
        print("loading from epoch: ", start_epoch, "best_sa=", best_sa)
        check_sparsity(model)
    else:
        all_result = {}
        all_result["train_ta"] = []
        all_result["test_ta"] = []
        all_result["val_ta"] = []

        start_epoch = 0
        start_state = 0

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
        acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        tacc = validate(val_loader, model, criterion, args)
        # evaluate on test set
        test_tacc = validate(test_loader, model, criterion, args)

        scheduler.step()

        all_result["train_ta"].append(acc)
        all_result["val_ta"].append(tacc)
        all_result["test_ta"].append(test_tacc)

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        save_checkpoint(
            {
                "state": state,
                "result": all_result,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_sa": best_sa,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "init_weight": initalization,
            },
            is_SA_best=is_best_sa,
            pruning=state,
            save_path=args.save_dir,
        )

        # plot training curve
        plt.plot(all_result["train_ta"], label="train_acc")
        plt.plot(all_result["val_ta"], label="val_acc")
        plt.plot(all_result["test_ta"], label="test_acc")
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, str(state) + "net_train.png"))
        plt.close()
        print("one epoch duration:{}".format(time.time() - start_time))

    # report result
    check_sparsity(model)
    print("Performance on the test data set")
    test_tacc = validate(test_loader, model, criterion, args)
    if len(all_result["val_ta"]) != 0:
        val_pick_best_epoch = np.argmax(np.array(all_result["val_ta"]))
        print(
            "* best SA = {}, Epoch = {}".format(
                all_result["test_ta"][val_pick_best_epoch], val_pick_best_epoch + 1
            )
        )

    all_result = {}
    all_result["train_ta"] = []
    all_result["test_ta"] = []
    all_result["val_ta"] = []
    best_sa = 0
    start_epoch = 0


if __name__ == "__main__":
    main()
