import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import src.pruner as pruner
from src.core import utils
from src.models import model_dict


class JsonSplitDataset(Dataset):
    def __init__(self, root, entries, transform=None):
        self.root = Path(root)
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, label = self.entries[index]
        image_path = Path(path)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)


def _setup_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        return torch.device(f"cuda:{int(args.gpu)}")
    return torch.device("cpu")


def _transfer_transform(args, train):
    if train and args.transfer_method == "ff":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(args.transfer_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    resize = int(round(args.transfer_resolution / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(args.transfer_resolution),
            transforms.ToTensor(),
        ]
    )


def _parse_split_entry(entry):
    if isinstance(entry, dict):
        path = entry.get("impath") or entry.get("image") or entry.get("path")
        label = entry.get("label")
    else:
        path, label = entry[0], entry[1]
    if path is None or label is None:
        raise ValueError(f"Unsupported split entry: {entry}")
    return path, int(label)


def _load_json_split(args, train_transform, test_transform):
    with open(args.transfer_split_file, "r") as f:
        split = json.load(f)
    train_entries = list(split.get("train", []))
    if args.transfer_include_val:
        train_entries.extend(split.get("val", []))
    test_entries = split.get("test", [])
    if not train_entries or not test_entries:
        raise ValueError("Split file must contain non-empty train and test entries")

    train_entries = [_parse_split_entry(entry) for entry in train_entries]
    test_entries = [_parse_split_entry(entry) for entry in test_entries]
    num_classes = max(label for _, label in train_entries + test_entries) + 1
    train_set = JsonSplitDataset(args.target_data, train_entries, train_transform)
    test_set = JsonSplitDataset(args.target_data, test_entries, test_transform)
    return train_set, test_set, num_classes


def _dataset_labels(dataset):
    for attr in ("targets", "labels", "_labels"):
        if hasattr(dataset, attr):
            return list(getattr(dataset, attr))
    if hasattr(dataset, "samples"):
        return [label for _, label in dataset.samples]
    raise AttributeError(f"Cannot find labels for {type(dataset).__name__}")


def _stratified_split(dataset, ratio, seed):
    labels = {}
    for index, label in enumerate(_dataset_labels(dataset)):
        labels.setdefault(int(label), []).append(index)

    rng = np.random.RandomState(seed)
    train_indices = []
    test_indices = []
    for indices in labels.values():
        rng.shuffle(indices)
        split = max(1, int(round(len(indices) * ratio)))
        split = min(split, len(indices) - 1) if len(indices) > 1 else len(indices)
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def _ensure_transfer_save_dir(args):
    if args.save_dir is None:
        args.save_dir = os.path.join(
            "./runs",
            "transfer",
            f"{args.target_dataset}_{args.arch}_{args.transfer_method}",
        )


def _build_transfer_datasets(args):
    train_transform = _transfer_transform(args, train=True)
    test_transform = _transfer_transform(args, train=False)

    if args.transfer_split_file is not None:
        return _load_json_split(args, train_transform, test_transform)

    if args.target_train_path and args.target_test_path:
        train_set = datasets.ImageFolder(args.target_train_path, transform=train_transform)
        test_set = datasets.ImageFolder(args.target_test_path, transform=test_transform)
        if train_set.class_to_idx != test_set.class_to_idx:
            raise ValueError("ImageFolder train/test class_to_idx mappings must match")
        return train_set, test_set, len(train_set.classes)

    target = args.target_dataset.lower()
    if target in {"oxfordpets", "oxford_iiit_pet", "pets"}:
        train_set = datasets.OxfordIIITPet(
            args.target_data,
            split="trainval",
            target_types="category",
            transform=train_transform,
            download=args.transfer_download,
        )
        test_set = datasets.OxfordIIITPet(
            args.target_data,
            split="test",
            target_types="category",
            transform=test_transform,
            download=args.transfer_download,
        )
        return train_set, test_set, 37

    if target == "sun397":
        full_train = datasets.SUN397(
            args.target_data, transform=train_transform, download=args.transfer_download
        )
        full_test = datasets.SUN397(
            args.target_data, transform=test_transform, download=False
        )
        print(
            "SUN397 has no train/test split in torchvision; using a deterministic "
            f"{args.transfer_train_ratio:.2f} stratified split. "
            "Pass --transfer_split_file for CoOp-style splits."
        )
        train_set, _ = _stratified_split(full_train, args.transfer_train_ratio, args.seed)
        _, test_set = _stratified_split(full_test, args.transfer_train_ratio, args.seed)
        return train_set, test_set, 397

    raise ValueError(
        "Unsupported target dataset. Use oxfordpets, sun397, imagefolder paths, "
        "or --transfer_split_file."
    )


def _clean_state_dict(state_dict):
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "state_dicts" in checkpoint and "network" in checkpoint["state_dicts"]:
            return checkpoint["state_dicts"]["network"]
    return checkpoint


def _replace_classifier(model, num_classes):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model.fc

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            in_features = classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            return model.classifier
        if isinstance(classifier, nn.Sequential):
            modules = list(classifier.children())
            for index in range(len(modules) - 1, -1, -1):
                if isinstance(modules[index], nn.Linear):
                    in_features = modules[index].in_features
                    modules[index] = nn.Linear(in_features, num_classes)
                    model.classifier = nn.Sequential(*modules)
                    return model.classifier[index]

    if hasattr(model, "mlp_head") and isinstance(model.mlp_head, nn.Sequential):
        modules = list(model.mlp_head.children())
        for index in range(len(modules) - 1, -1, -1):
            if isinstance(modules[index], nn.Linear):
                in_features = modules[index].in_features
                modules[index] = nn.Linear(in_features, num_classes)
                model.mlp_head = nn.Sequential(*modules)
                return model.mlp_head[index]

    raise ValueError("Could not find a replaceable classifier head on the model")


def _build_model(args, num_classes, device):
    if args.source_checkpoint is None and not args.transfer_eval_only:
        raise ValueError("--source_checkpoint is required for transfer learning")

    model = model_dict[args.arch](
        num_classes=args.source_num_classes, imagenet=True
    )
    model.normalize = utils.NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if args.source_checkpoint is not None:
        checkpoint = torch.load(args.source_checkpoint, map_location=device)
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        current_mask = pruner.extract_mask(state_dict)
        if current_mask:
            pruner.prune_model_custom(model, current_mask)
            pruner.check_sparsity(model)
        model.load_state_dict(state_dict, strict=False)

    head = _replace_classifier(model, num_classes)
    model.to(device)

    if args.transfer_method == "lp":
        model.requires_grad_(False)
        head.requires_grad_(True)
    return model


def _build_optimizer(args, model):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found for transfer learning")
    if args.transfer_optimizer == "SGD":
        return torch.optim.SGD(
            params,
            lr=args.transfer_lr,
            momentum=args.momentum,
            weight_decay=args.transfer_weight_decay,
        )
    return torch.optim.Adam(
        params, lr=args.transfer_lr, weight_decay=args.transfer_weight_decay
    )


def _accuracy(output, target, topk=(1,)):
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result


def _run_epoch(loader, model, criterion, optimizer, scheduler, args, device, epoch):
    if args.transfer_method == "lp":
        model.eval()
    else:
        model.train()

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    start = time.time()
    for step, (image, target) in enumerate(loader):
        image = image.to(device)
        target = target.to(device)

        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc1 = _accuracy(output.float(), target, topk=(1,))[0]
        losses.update(loss.item(), image.size(0))
        top1.update(acc1.item(), image.size(0))

        if (step + 1) % args.print_freq == 0:
            print(
                "Transfer epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, step, len(loader), time.time() - start, loss=losses, top1=top1
                )
            )
            start = time.time()
    return {"loss": losses.avg, "top1": top1.avg}


def _evaluate(loader, model, criterion, device):
    model.eval()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    with torch.no_grad():
        for image, target in loader:
            image = image.to(device)
            target = target.to(device)
            output = model(image)
            loss = criterion(output, target)
            acc1, acc5 = _accuracy(output.float(), target, topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1.update(acc1.item(), image.size(0))
            top5.update(acc5.item(), image.size(0))
    return {"loss": losses.avg, "top1": top1.avg, "top5": top5.avg}


def _save_transfer_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, result):
    os.makedirs(path, exist_ok=True)
    state = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_acc": best_acc,
        "result": result,
    }
    torch.save(state, os.path.join(path, "transfer_checkpoint.pth.tar"))
    torch.save(result, os.path.join(path, "transfer_result.pth.tar"))


def run_transfer(args):
    device = _setup_device(args)
    _ensure_transfer_save_dir(args)
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)

    train_set, test_set, num_classes = _build_transfer_datasets(args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.transfer_test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    target_name = args.target_dataset
    if args.target_train_path and args.target_test_path:
        target_name = "imagefolder"
    if args.transfer_split_file is not None:
        target_name = f"split:{args.transfer_split_file}"

    print(
        "Transfer dataset: "
        f"{target_name}, train={len(train_set)}, test={len(test_set)}, "
        f"classes={num_classes}"
    )

    model = _build_model(args, num_classes, device)
    criterion = nn.CrossEntropyLoss()

    result = {"train": [], "test": []}
    best_acc = 0

    if args.transfer_eval_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required with --transfer_eval_only")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = _clean_state_dict(_extract_state_dict(checkpoint))
        current_mask = pruner.extract_mask(state_dict)
        if current_mask:
            pruner.prune_model_custom(model, current_mask)
        model.load_state_dict(state_dict, strict=False)
        test_stats = _evaluate(test_loader, model, criterion, device)
        result["test"].append(test_stats)
        print(f"transfer test acc: {test_stats['top1']:.3f}")
        _save_transfer_checkpoint(args.save_dir, model, None, None, 0, test_stats["top1"], result)
        return

    optimizer = _build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.transfer_epochs * len(train_loader))
    )

    for epoch in range(args.transfer_epochs):
        start = time.time()
        train_stats = _run_epoch(
            train_loader, model, criterion, optimizer, scheduler, args, device, epoch
        )
        test_stats = _evaluate(test_loader, model, criterion, device)
        result["train"].append(train_stats)
        result["test"].append(test_stats)
        best_acc = max(best_acc, test_stats["top1"])
        print(
            "Transfer epoch #{}, train_acc={:.3f}, test_acc={:.3f}, "
            "test_top5={:.3f}, best_acc={:.3f}, duration={:.2f}".format(
                epoch,
                train_stats["top1"],
                test_stats["top1"],
                test_stats["top5"],
                best_acc,
                time.time() - start,
            )
        )
        _save_transfer_checkpoint(
            args.save_dir, model, optimizer, scheduler, epoch + 1, best_acc, result
        )
