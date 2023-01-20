import pruner
import trainer
from .FT import FT_l1


def FT_prune(data_loaders, model, criterion, args):
    test_loader = data_loaders["test"]

    # save checkpoint
    initialization = model.state_dict()

    # unlearn
    FT_l1(data_loaders, model, criterion, args)

    # val
    pruner.check_sparsity(model)
    trainer.validate(test_loader, model, criterion, args)

    # prune
    if args.random_prune:
        print('random pruning')
        pruner.pruning_model_random(model, args.rate)
    else:
        print('L1 pruning')
        pruner.pruning_model(model, args.rate)

    pruner.check_sparsity(model)

    # recover state_dict
    current_mask = pruner.extract_mask(model.state_dict())
    pruner.remove_prune(model)
    model.load_state_dict(initialization, strict=False)
    pruner.prune_model_custom(model, current_mask)

    # unlearn
    FT_l1(data_loaders, model, criterion, args)

    return model
