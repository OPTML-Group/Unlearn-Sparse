import torch
import time
from copy import deepcopy
import utils
from .impl import iterative_unlearn
import numpy as np

@iterative_unlearn
def RL(data_loaders, model, criterion, optimizer, epoch, args):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
    
    retain_dataset = retain_loader.dataset
    train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    loader_len = len(forget_loader) + len(retain_loader)

    if epoch < args.warmup:
        utils.warmup_lr(epoch, i+1, optimizer,
                        one_epoch_step=loader_len, args=args)

    # for i, (image, target) in enumerate(forget_loader):
    #     image = image.cuda()
    #     target = torch.randint(0, args.num_classes, target.shape).cuda()

    #     # compute output
    #     output_clean = model(image)
    #     loss = criterion(output_clean, target)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     output = output_clean.float()
    #     loss = loss.float()
    #     # measure accuracy and record loss
    #     prec1 = utils.accuracy(output.data, target)[0]

    #     losses.update(loss.item(), image.size(0))
    #     top1.update(prec1.item(), image.size(0))

    #     if (i + 1) % args.print_freq == 0:
    #         end = time.time()
    #         print('Epoch: [{0}][{1}/{2}]\t'
    #               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #               'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
    #               'Time {3:.2f}'.format(
    #                   epoch, i, loader_len, end-start, loss=losses, top1=top1))
    #         start = time.time()

    for it, (image, target) in enumerate(train_loader):
        i = it + len(forget_loader)
        image = image.cuda()
        target = target.cuda()
        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Time {3:.2f}'.format(
                      epoch, i, loader_len, end-start, loss=losses, top1=top1))
            start = time.time()


    return top1.avg
