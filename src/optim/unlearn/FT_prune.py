import src.pruner as pruner
import src.trainer as trainer

from .FT import FT_l1


def FT_prune(data_loaders, model, criterion, args):
    test_loader = data_loaders["test"]

    # unlearn
    FT_l1(data_loaders, model, criterion, args)

    # val
    pruner.check_sparsity(model)
    trainer.validate(test_loader, model, criterion, args)

    return model
