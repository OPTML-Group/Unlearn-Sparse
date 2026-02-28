from optim.LS import LabelSmoothingCrossEntropy
from trainer import train

from .impl import iterative_unlearn


@iterative_unlearn
def retrain_ls(data_loaders, model, criterion, optimizer, epoch, args):
    retain_loader = data_loaders["retain"]
    return train(
        retain_loader, model, LabelSmoothingCrossEntropy(), optimizer, epoch, args
    )
