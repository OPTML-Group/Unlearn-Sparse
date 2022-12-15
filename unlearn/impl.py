import torch
import time
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

import pruner
from trainer import validate
import utils


def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f'{name}_acc')
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + '_train.png'))
    plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args):
    state = {
        'state_dict': model.state_dict(),
        'evaluation_result': evaluation_result
    }
    utils.save_checkpoint(state, False, args.save_dir, args.unlearn)
    utils.save_checkpoint(evaluation_result, False, args.save_dir,
                          args.unlearn, filename="eval_result.pth.tar")


def load_unlearn_checkpoint(model, device, args):
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
    if checkpoint is None or checkpoint.get('state_dict') is None:
        return None

    current_mask = pruner.extract_mask(checkpoint['state_dict'])
    pruner.prune_model_custom(model, current_mask)
    pruner.check_sparsity(model)

    model.load_state_dict(checkpoint['state_dict'], strict=False)

    # adding an extra forward process to enable the masks
    x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
    model.eval()
    with torch.no_grad():
        model(x_rand)

    evaluation_result = checkpoint.get('evaluation_result')
    return model, evaluation_result


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args):
        decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=decreasing_lr, gamma=0.1)  # 0.1 is fixed

        # results = OrderedDict((name, []) for name in data_loaders.keys())
        # results['train'] = []

        for epoch in range(0, args.epochs):
            start_time = time.time()
            print("Epoch #{}, Learning rate: {}".format(
                epoch, optimizer.state_dict()['param_groups'][0]['lr']))
            train_acc = unlearn_iter_func(
                data_loaders, model, criterion, optimizer, epoch, args)
            scheduler.step()

            # results['train'].append(train_acc)
            # for name, loader in data_loaders.items():
            #     print(f"{name} acc:")
            #     val_acc = validate(loader, model, criterion, args)
            #     results[name].append(val_acc)

            # plot_training_curve(results, args.save_dir, args.unlearn)

            print("one epoch duration:{}".format(time.time()-start_time))

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)
