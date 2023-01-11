'''
    setup model and datasets
'''


import time
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
from torchvision import transforms

__all__ = ['setup_model_dataset', 'AverageMeter',
           'warmup_lr', 'save_checkpoint', 'setup_seed', 'accuracy']


def warmup_lr(epoch, step, optimizer, one_epoch_step, args):

    overall_steps = args.warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step

    lr = args.lr * current_steps/overall_steps
    lr = min(lr, args.lr)

    for p in optimizer.param_groups:
        p['lr'] = lr


def save_checkpoint(state, is_SA_best, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning)+filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(
            save_path, str(pruning)+'model_SA_best.pth.tar'))


def load_checkpoint(device, save_path, pruning, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, str(pruning)+filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None


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


def dataset_convert_to_test(dataset):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    dataset.transform = test_transform
    dataset.train = False


def setup_model_dataset(args):

    if args.dataset == 'cifar10':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_full_loader, val_loader, test_loader = cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
        marked_loader, _, _ = cifar10_dataloaders(batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers, class_to_replace=args.class_to_replace,
                                                  num_indexes_to_replace=args.num_indexes_to_replace, indexes_to_replace=args.indexes_to_replace, seed=args.seed, only_mark=True, shuffle=True, no_aug=args.no_aug)
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        print(model)
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == 'cifar100':
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_full_loader, val_loader, test_loader = cifar100_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
        marked_loader, _, _ = cifar100_dataloaders(batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers, class_to_replace=args.class_to_replace,
                                                  num_indexes_to_replace=args.num_indexes_to_replace, indexes_to_replace=args.indexes_to_replace, seed=args.seed, only_mark=True, shuffle=True, no_aug=args.no_aug)        
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)
        model.normalize = normalization
        print(model)
        return model, train_full_loader, val_loader, test_loader, marked_loader
    elif args.dataset == 'TinyImagenet':
        classes = 200
        normalization = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_full_loader, val_loader, test_loader = TinyImageNet(
            args).data_loaders(batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
        # train_full_loader, val_loader, test_loader =None, None,None
        marked_loader, _,_ = TinyImageNet(
            args).data_loaders(batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers, class_to_replace=args.class_to_replace,
                                                  num_indexes_to_replace=args.num_indexes_to_replace, indexes_to_replace=args.indexes_to_replace, seed=args.seed, only_mark=True, shuffle=True)
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        print(model)
        return model, train_full_loader, val_loader, test_loader,marked_loader
    

    elif args.dataset == 'cifar100_no_val':
        classes = 100
        normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2673,	0.2564,	0.2762])
        train_set_loader, val_loader, test_loader = cifar100_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)

    elif args.dataset == 'cifar10_no_val':
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_set_loader, val_loader, test_loader = cifar10_dataloaders_no_val(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)

    elif args.dataset == "fast_cifar":
        classes = 10
        normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_full_loader, val_loader, test_loader = fast_cifar10_dataloaders(
            batch_size=args.batch_size, data_dir=args.data, num_workers=args.workers)
        # marked_loader, _,_ = cifar10_dataloaders(batch_size = args.batch_size, data_dir = args.data, num_workers = args.workers,class_to_replace=args.class_to_replace,num_indexes_to_replace=args.num_indexes_to_replace,indexes_to_replace=args.indexes_to_replace, seed=args.seed, only_mark= True,shuffle = True)
        if args.imagenet_arch:
            model = model_dict[args.arch](num_classes=classes, imagenet=True)
        else:
            model = model_dict[args.arch](num_classes=classes)

        model.normalize = normalization
        print(model)
        return model, train_full_loader, val_loader, test_loader

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def run_commands(gpus, commands, call=False, dir="commands", shuffle=True, delay=0.5):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(dir, exist_ok=True)

    fout = open('stop_{}.sh'.format(dir), 'w')
    print("kill $(ps aux|grep 'bash " + dir + "'|awk '{print $2}')", file=fout)
    fout.close()

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)

        sh_path = os.path.join(dir, "run{}.sh".format(i))
        fout = open(sh_path, 'w')
        for com in i_commands:
            print(prefix + com, file=fout)
        fout.close()
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)
