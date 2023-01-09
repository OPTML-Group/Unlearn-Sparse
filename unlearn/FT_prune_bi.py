import time
import torch
import sys
sys.path.append("/mnt/home/jiajingh/Unlearn-Sparse/trainer")
sys.path.append("/mnt/home/jiajingh/Unlearn-Sparse/pruner")
sys.path.append("/mnt/home/jiajingh/Unlearn-Sparse/")
from .impl import iterative_unlearn
from copy import deepcopy
from pruner import *
from trainer import train, validate
import matplotlib.pyplot as plt
import os
from utils import *
import numpy as np
def FT_prune_bi(data_loaders, model, criterion, args):
    best_sa = 0
    train_loader = data_loaders["retain"]
    test_loader = data_loaders["test"]
    val_loader = data_loaders["val"]
    all_result = {}
    all_result['train_ta'] = []
    all_result['test_ta'] = []
    all_result['val_ta'] = []
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
    start_epoch = 0
    start_state = 0
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1)
    print('######################################## Start Standard Training Iterative Pruning ########################################')

    # for state in range(start_state, args.pruning_times):

    #     print('******************************************')
    #     print('pruning state', state)
    #     print('******************************************')

    check_sparsity(model)
    state = 0
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        # if state == 0:
        #     if (epoch) == args.rewind_epoch:
        #         torch.save(model.state_dict(), os.path.join(
        #             args.save_dir, 'epoch_{}_rewind_weight.pt'.format(epoch+1)))
        #         if args.prune_type == 'rewind_lt':
        #             initalization = deepcopy(model.state_dict())
        acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        tacc = validate(val_loader, model, criterion, args)
        # evaluate on test set
        test_tacc = validate(test_loader, model, criterion, args)

        scheduler.step()

        all_result['train_ta'].append(acc)
        all_result['val_ta'].append(tacc)
        all_result['test_ta'].append(test_tacc)

        # remember best prec@1 and save checkpoint
        is_best_sa = tacc > best_sa
        best_sa = max(tacc, best_sa)

        save_checkpoint({
            'state': state,
            'result': all_result,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_sa': best_sa,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'init_weight': None
        }, is_SA_best=is_best_sa, pruning=state, save_path=args.save_dir)

        # plot training curve
        plt.plot(all_result['train_ta'], label='train_acc')
        plt.plot(all_result['val_ta'], label='val_acc')
        plt.plot(all_result['test_ta'], label='test_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir,
                    str(state)+'net_train.png'))
        plt.close()
        print("one epoch duration:{}".format(time.time()-start_time))

        # report result
        check_sparsity(model)
        print("Performance on the test data set")
        test_tacc = validate(test_loader, model, criterion, args)
        if len(all_result['val_ta']) != 0:
            val_pick_best_epoch = np.argmax(np.array(all_result['val_ta']))
            print('* best SA = {}, Epoch = {}'.format(
                all_result['test_ta'][val_pick_best_epoch], val_pick_best_epoch+1))
        #pruning and rewind
        if args.random_prune:
            print('random pruning')
            pruning_model_random(model, args.rate)
        else:
            print('L1 pruning')
            pruning_model(model, args.rate)

        remain_weight = check_sparsity(model)

        # weight rewinding
        # rewind, initialization is a full model architecture without masks

    return model