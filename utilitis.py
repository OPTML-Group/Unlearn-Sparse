'''
    setup model and datasets
'''



import os
import copy 
import torch
import numpy as np
from dataset import TinyImageNet 
# from advertorch.utils import NormalizeByChannelMeanStd
import shutil
from models import *
from dataset import *
import random

__all__ = ['setup_model_dataset','AverageMeter','warmup_lr','save_checkpoint','setup_seed']

def warmup_lr(epoch, step, optimizer, one_epoch_step,args):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr']=lr

def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning)+filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, str(pruning)+'model_SA_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_model_dataset(args):

    if args.dataset == 'cifar10':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_full_loader, val_loader, test_loader = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)
        marked_loader, _,_ = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers,class_to_replace=args.class_to_replace,num_indexes_to_replace=args.num_indexes_to_replace,indexes_to_replace=args.indexes_to_replace, seed=args.seed, only_mark= True,shuffle = True)
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        print(model)
        return model,train_full_loader,val_loader,test_loader,marked_loader
    elif args.dataset == 'cifar100':
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_set_loader, val_loader, test_loader = cifar100_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)
    
    elif args.dataset == 'TinyImagenet':
        classes = 200
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_set_loader, val_loader, test_loader = TinyImageNet(args).data_loaders()

    elif args.dataset == 'cifar100_no_val':
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_set_loader, val_loader, test_loader = cifar100_dataloaders_no_val(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)
    
    elif args.dataset == 'cifar10_no_val':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, val_loader, test_loader = cifar10_dataloaders_no_val(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers)
    


    else:
        raise ValueError('Dataset not supprot yet !')
    # import pdb;pdb.set_trace()

    if args.imagenet_arch:
        model = model_dict[args.arch](num_classes=classes, imagenet=True)
    else:
        model = model_dict[args.arch](num_classes=classes)

    model.normalize = normalization
    print(model)

    return model, train_set_loader, val_loader, test_loader

def setup_seed(seed): 
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return self.normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

    def normalize_fn(self, tensor, mean, std):
        """Differentiable version of torchvision.functional.normalize"""
        # here we assume the color channel is in at dim=1
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        return tensor.sub(mean).div(std)
