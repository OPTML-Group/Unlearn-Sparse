'''
    main process for a Lottery Tickets experiments
'''
import os
import pdb
import time 
import pickle
import random
import shutil
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt
import copy
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

import utils
from pruner import *

import arg_parser

best_sa = 0

def main():
    global args, best_sa
    args = arg_parser.parse_args()
    print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset 
    model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
    model.cuda()
    def replace_loader_dataset(data_loader, dataset, batch_size=args.batch_size, seed=1, shuffle=True):
        utils.setup_seed(seed)
        loader_args = {'num_workers': 0, 'pin_memory': False}
        def _init_fn(worker_id):
            np.random.seed(int(seed))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=0,pin_memory=True,shuffle=shuffle)

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    marked = forget_dataset.targets < 0
    forget_dataset.data = forget_dataset.data[marked]
    forget_dataset.targets = - forget_dataset.targets[marked] - 1
    forget_loader = replace_loader_dataset(train_loader_full, forget_dataset, seed=seed, shuffle=True)
    print(len(forget_dataset))
    retain_dataset = copy.deepcopy(marked_loader.dataset)
    marked = retain_dataset.targets >= 0
    retain_dataset.data = retain_dataset.data[marked]
    retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = replace_loader_dataset(train_loader_full, retain_dataset, seed=seed, shuffle=True)
    print(len(retain_dataset))
    assert(len(forget_dataset) + len(retain_dataset) == len(train_loader_full.dataset))

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1) # 0.1 is fixed

    if args.resume:
        print('resume from checkpoint {}'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location = torch.device('cuda:'+str(args.gpu)))
        best_sa = checkpoint['best_sa']
        start_epoch = checkpoint['epoch']
        all_result = checkpoint['result']
        start_state = checkpoint['state']

        if start_state > 0:
            current_mask = extract_mask(checkpoint['state_dict'])
            prune_model_custom(model, current_mask)
            check_sparsity(model)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # adding an extra forward process to enable the masks
        x_rand = torch.rand(1,3,args.input_size, args.input_size).cuda()
        model.eval()
        with torch.no_grad():
            model(x_rand)

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        initalization = checkpoint['init_weight']
        print('loading state:', start_state)
        print('loading from epoch: ',start_epoch, 'best_sa=', best_sa)

    else:
        if args.unlearn == "retrain":
            all_result = {}
            all_result['retain_ta'] = []
            all_result['test_ta'] = []
            all_result['val_ta'] = []
            all_result['forget_ta'] = []
            start_epoch = 0
            start_state = 0
            checkpoint = torch.load(args.mask, map_location = torch.device('cuda:'+str(args.gpu)))
            current_mask = extract_mask(checkpoint)
            prune_model_custom(model, current_mask)
            check_sparsity(model)
            test_tacc = validate(test_loader, model, criterion)
            check_sparsity(model)        
            for epoch in range(0, args.epochs):
                start_time = time.time()
                print(optimizer.state_dict()['param_groups'][0]['lr'])
                acc = train(retain_loader, model, criterion, optimizer, epoch)

                # evaluate on validation set
                tacc = validate(val_loader, model, criterion)
                # evaluate on test set
                test_tacc = validate(test_loader, model, criterion)
                # evaluate on forget set
                f_tacc = validate(forget_loader, model, criterion)
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
                plt.plot(all_result['retain'], label='train_acc')
                plt.plot(all_result['val_ta'], label='val_acc')
                plt.plot(all_result['test_ta'], label='test_acc')
                plt.plot(all_result['forget_ta'], label='forget_acc')
                plt.legend()
                plt.savefig(os.path.join(args.save_dir, str(0)+'net_train.png'))
                plt.close()
                print("one epoch duration:{}".format(time.time()-start_time))
        else:
            checkpoint = torch.load(args.mask, map_location = torch.device('cuda:'+str(args.gpu)))
            current_mask = extract_mask(checkpoint)
            prune_model_custom(model, current_mask)
            check_sparsity(model)
            model.load_state_dict(checkpoint, strict=False)
            # test_tacc = validate(test_loader, model, criterion)


        if args.unlearn == "RL":
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
                tacc = validate(val_loader, model, criterion)
                acc = RL(forget_loader, model, criterion, optimizer, epoch)

                # evaluate on validation set
                tacc = validate(val_loader, model, criterion)
                # evaluate on test set
                test_tacc = validate(test_loader, model, criterion)
                # evaluate on forget set
                f_tacc = validate(retain_loader, model, criterion)
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
                plt.plot(all_result['retain_ta'], label='train_acc')
                plt.plot(all_result['val_ta'], label='val_acc')
                plt.plot(all_result['test_ta'], label='test_acc')
                plt.plot(all_result['forget_ta'], label='forget_acc')
                plt.legend()
                plt.savefig(os.path.join(args.save_dir, str(0)+'net_train.png'))
                plt.close()
                print("one epoch duration:{}".format(time.time()-start_time))
        elif args.unlearn == "GA":
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
                acc = GA(forget_loader, model, criterion, optimizer, epoch)

                # evaluate on validation set
                tacc = validate(val_loader, model, criterion)
                # evaluate on test set
                test_tacc = validate(test_loader, model, criterion)
                # evaluate on retain set
                f_tacc = validate(retain_loader, model, criterion)
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



        




def train(train_loader, model, criterion, optimizer, epoch):
    
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


def GA(train_loader, model, criterion, optimizer, epoch):
    
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


def RL(train_loader, model, criterion, optimizer, epoch):
    
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader), args=args)

        image = image.cuda()
        target = torch.randint(0,9,target.shape)
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

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg

if __name__ == '__main__':
    main()
