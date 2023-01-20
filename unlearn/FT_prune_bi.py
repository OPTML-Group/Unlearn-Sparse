import time
import pruner
from trainer import train, validate, get_optimizer_and_scheduler


def FT_prune_bi(data_loaders, model, criterion, args):
    train_loader = data_loaders["retain"]
    test_loader = data_loaders["test"]

    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    prune_rate = args.rate ** (1 / (args.unlearn_epochs - 1))

    pruner.check_sparsity(model)

    for epoch in range(args.unlearn_epochs):
        start_time = time.time()

        # training
        train(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step()

        print("training duration:{}".format(time.time()-start_time))

        # eval
        pruner.check_sparsity(model)
        validate(test_loader, model, criterion, args)

        # pruning
        if epoch < args.unlearn_epochs - 1:
            if args.random_prune:
                print('random pruning')
                pruner.pruning_model_random(model, prune_rate)
            else:
                print('L1 pruning')
                pruner.pruning_model(model, prune_rate)

        print("one epoch duration:{}".format(time.time()-start_time))

    pruner.check_sparsity(model)

    return model
