import torch
import time
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

from trainer import validate
import utils

def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args):
        decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1) # 0.1 is fixed

        results = OrderedDict((name, []) for name in data_loaders.keys())
        train_result = []

        for epoch in range(0, args.epochs):
            start_time = time.time()
            print("Epoch #{}, Learning rate: {}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
            train_acc = unlearn_iter_func(data_loaders, model, criterion, optimizer, epoch, args)
            train_result.append(train_acc)

            for name, loader in data_loaders.items():
                print(f"{name} acc:")
                val_acc = validate(loader, model, criterion, args)
                results[name].append(val_acc)
            scheduler.step()

            utils.save_checkpoint({
                'state': 0,
                'result': results,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'init_weight': None
            }, is_SA_best=True, pruning=0, save_path=args.save_dir)

            # plot training curve
            for name, result in results.items():
                plt.plot(result, label=f'{name}_acc')
            plt.plot(train_result, label = 'train_acc')
            plt.legend()
            plt.savefig(os.path.join(args.save_dir, str(0)+'net_train.png'))
            plt.close()

            print("one epoch duration:{}".format(time.time()-start_time))

    return _wrapped


def iterative_unlearn(func):
    """usage:
    
    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)