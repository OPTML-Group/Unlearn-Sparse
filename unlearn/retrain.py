import torch
import time
import os
import matplotlib.pyplot as plt

from pruner import extract_mask, check_sparsity, prune_model_custom
from trainer import validate, train
import utils

def retrain(forget_loader, retain_loader, test_loader, val_loader, model, criterion, optimizer, scheduler, args):
    all_result = {}
    all_result['retain_ta'] = []
    all_result['test_ta'] = []
    all_result['val_ta'] = []
    all_result['forget_ta'] = []
    start_epoch = 0
    start_state = 0
    best_sa = 0
    checkpoint = torch.load(args.mask, map_location = torch.device('cuda:'+str(args.gpu)))
    current_mask = extract_mask(checkpoint)
    prune_model_custom(model, current_mask)
    check_sparsity(model)
    test_tacc = validate(test_loader, model, criterion, args)
    check_sparsity(model)

    for epoch in range(0, args.epochs):
        start_time = time.time()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc = train(retain_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        tacc = validate(val_loader, model, criterion, args)
        # evaluate on test set
        test_tacc = validate(test_loader, model, criterion, args)
        # evaluate on forget set
        f_tacc = validate(forget_loader, model, criterion, args)
        scheduler.step()

        all_result['retain_ta'].append(acc)
        all_result['test_ta'].append(tacc)
        all_result['val_ta'].append(test_tacc)
        all_result['forget_ta'].append(f_tacc)
        # remember best prec@1 and save checkpoint
        is_best_sa = tacc  > best_sa
        best_sa = max(tacc, best_sa)

        utils.save_checkpoint({
            'state': 0,
            'result': all_result,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_sa': best_sa,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'init_weight': None
        }, is_SA_best=is_best_sa, pruning=0, save_path=args.save_dir)

        # plot training curve
        plt.plot(all_result['retain_ta'], label='train_acc')
        plt.plot(all_result['val_ta'], label='val_acc')
        plt.plot(all_result['test_ta'], label='test_acc')
        plt.plot(all_result['forget_ta'], label='forget_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, str(0)+'net_train.png'))
        plt.close()
        print("one epoch duration:{}".format(time.time()-start_time))