import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.data

import evaluation
import pruner
import unlearn
import utils
from trainer import validate


def _setup_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        return torch.device(f"cuda:{int(args.gpu)}")
    return torch.device("cpu")


def _prepare_loaders(args, *, imagenet=False):
    if imagenet:
        model, retain_loader, forget_loader, val_loader = utils.setup_model_dataset(args)
        print(len(retain_loader.dataset))
        print(len(forget_loader.dataset))
        loaders = OrderedDict(
            retain=retain_loader, forget=forget_loader, val=val_loader, test=val_loader
        )
        return model, loaders, retain_loader.dataset, forget_loader.dataset

    model, train_loader_full, val_loader, test_loader, marked_loader = (
        utils.setup_model_dataset(args)
    )
    forget_loader, retain_loader = utils.get_unlearn_loader(marked_loader, args)
    forget_dataset = forget_loader.dataset
    retain_dataset = retain_loader.dataset
    assert len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset)

    loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )
    return model, loaders, retain_dataset, forget_dataset


def _load_mask_and_unlearn(model, device, args, loaders, imagenet=False):
    evaluation_result = None
    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)
    else:
        checkpoint = None

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        current_mask = pruner.extract_mask(checkpoint)
        pruner.prune_model_custom(model, current_mask)
        pruner.check_sparsity(model)

        if args.unlearn != "retrain":
            if not imagenet and args.unlearn in {"retrain_sam", "retrain_ls"}:
                pass
            else:
                model.load_state_dict(checkpoint, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        criterion = nn.CrossEntropyLoss()
        unlearn_method(loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}
    return model, evaluation_result


def _evaluate_accuracy(
    model, loaders, args, evaluation_result, guard_key, reverse=False, convert_to_test=False
):
    if guard_key in evaluation_result or "accuracy" in evaluation_result:
        return
    ordered = dict(reversed(list(loaders.items()))) if reverse else loaders
    accuracy = {}
    criterion = nn.CrossEntropyLoss()
    for name, loader in ordered.items():
        print("start testing")
        if convert_to_test:
            utils.dataset_convert_to_test(loader.dataset, args)
        val_acc = validate(loader, model, criterion, args)
        accuracy[name] = val_acc
        print(f"{name} acc: {val_acc}")

    evaluation_result["accuracy"] = accuracy
    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


def _evaluate_non_imagenet_mia(model, loaders, retain_dataset, forget_loader, args, evaluation_result):
    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    test_loader = loaders["test"]
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader.dataset, args)
        utils.dataset_convert_to_test(test_loader.dataset, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    if "SVC_MIA_training_privacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        retain_len = len(retain_dataset)
        num = test_len // 2

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader.dataset, args)
        utils.dataset_convert_to_test(test_loader.dataset, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(num)))
        target_train = torch.utils.data.Subset(retain_dataset, list(range(num, retain_len)))
        shadow_test = torch.utils.data.Subset(test_loader.dataset, list(range(num)))
        target_test = torch.utils.data.Subset(test_loader.dataset, list(range(num, test_len)))

        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=args.batch_size, shuffle=False
        )
        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=args.batch_size, shuffle=False
        )
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_training_privacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=shadow_test_loader,
            target_train=target_train_loader,
            target_test=target_test_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


def _evaluate_imagenet_mia(model, loaders, retain_dataset, forget_loader, args, evaluation_result):
    if "SVC_MIA_forget_efficacy" in evaluation_result:
        return

    val_loader = loaders["val"]
    n = min(10000, len(val_loader.dataset), len(retain_dataset))
    print(len(val_loader.dataset))

    val_subset = torch.utils.data.Subset(val_loader.dataset, list(range(n)))
    shadow_train = torch.utils.data.Subset(retain_dataset, list(range(n)))

    shadow_train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=args.batch_size, shuffle=False
    )
    shadow_test_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False
    )

    evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
        shadow_train=shadow_train_loader,
        shadow_test=shadow_test_loader,
        target_train=None,
        target_test=forget_loader,
        model=model,
    )
    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


def run_forget(args, *, imagenet=False):
    device = _setup_device(args)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)

    model, loaders, retain_dataset, forget_dataset = _prepare_loaders(
        args, imagenet=imagenet
    )
    model.cuda()

    model, evaluation_result = _load_mask_and_unlearn(
        model, device, args, loaders, imagenet=imagenet
    )

    if imagenet:
        _evaluate_accuracy(
            model,
            loaders,
            args,
            evaluation_result,
            guard_key="accuracy",
            reverse=True,
        )
        _evaluate_imagenet_mia(
            model, loaders, retain_dataset, loaders["forget"], args, evaluation_result
        )
    else:
        _evaluate_accuracy(
            model,
            loaders,
            args,
            evaluation_result,
            guard_key="new_accuracy",
            reverse=False,
            convert_to_test=True,
        )
        _evaluate_non_imagenet_mia(
            model, loaders, retain_dataset, loaders["forget"], args, evaluation_result
        )

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
