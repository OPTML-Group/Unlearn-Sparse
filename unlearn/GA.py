import time
import matplotlib.pyplot as plt
import os

from trainer import validate
import utils

def GA_iter(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader), args=args)

        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        loss = -criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def GA(forget_loader, retain_loader, test_loader, val_loader, model, criterion, optimizer, scheduler, args):
    all_result = {}
    all_result['retain_ta'] = []
    all_result['test_ta'] = []
    all_result['val_ta'] = []
    all_result['forget_ta'] = []
    start_epoch = 0
    start_state = 0
    for epoch in range(0, args.epochs):
        start_time = time.time()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc = GA_iter(forget_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        tacc = validate(val_loader, model, criterion, args)
        # evaluate on test set
        test_tacc = validate(test_loader, model, criterion, args)
        # evaluate on retain set
        f_tacc = validate(retain_loader, model, criterion, args)
        scheduler.step()

        all_result['retain_ta'].append(f_tacc)
        all_result['test_ta'].append(tacc)
        all_result['val_ta'].append(test_tacc)
        all_result['forget_ta'].append(acc)
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
        plt.plot(all_result['retain'], label='train_acc')
        plt.plot(all_result['val_ta'], label='val_acc')
        plt.plot(all_result['test_ta'], label='test_acc')
        plt.plot(all_result['forget_ta'], label='forget_acc')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, str(0)+'net_train.png'))
        plt.close()
        print("one epoch duration:{}".format(time.time()-start_time))