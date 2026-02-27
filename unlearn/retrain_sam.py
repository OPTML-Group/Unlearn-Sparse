import torch

from trainer import train_sam_epoch

from .impl import iterative_unlearn


@iterative_unlearn
def retrain_sam(data_loaders, model, criterion, optimizer, epoch, args):
    base_optimizer = torch.optim.SGD
    from SAM import SAM

    sam_optimizer = SAM(
        model.parameters(),
        base_optimizer,
        rho=2.0,
        adaptive=True,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    retain_loader = data_loaders["retain"]
    return train_sam_epoch(retain_loader, model, criterion, sam_optimizer, epoch, args)
