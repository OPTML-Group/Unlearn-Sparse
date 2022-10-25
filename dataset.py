'''
    function for loading datasets
    contains: 
        CIFAR-10
        CIFAR-100   
'''
import time
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import glob
import copy
from shutil import move

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders', 'cifar10_dataloaders_no_val',
           'cifar100_dataloaders_no_val', 'fast_cifar10_dataloaders']


def make_dataloaders_cifar(train_dataset=None, val_dataset=None, test_dataset=None, batch_size=None, num_workers=None):
    paths = {
        'train': train_dataset,
        'test': test_dataset,
        'val': val_dataset
    }

    start_time = time.time()
    # CIFAR_MEAN = [125.307, 122.961, 113.8575]
    # CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ['train', 'test', 'val']:
        label_pipeline: List[Operation] = [
            IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip()
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16)
        ])

        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name],
                               batch_size=batch_size,
                               num_workers=8,
                               order=OrderOption.RANDOM,
                               drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline,
                                          'label': label_pipeline})

    return loaders, start_time


def fast_cifar10_dataloaders(batch_size=128, data_dir='datasets/cifar10', num_workers=2, class_to_replace: int = None, num_indexes_to_replace=None, indexes_to_replace=None, seed: int = 1, only_mark: bool = False, shuffle=True):

    print('Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = CIFAR10(data_dir, train=True,  download=True)

    test_set = CIFAR10(data_dir, train=False,  download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(rng.choice(class_idx, int(
            0.1*len(class_idx)), replace=False))
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set)))-set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified")
    if class_to_replace is not None:
        replace_class(train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace, seed=seed-1,
                      only_mark=only_mark)
        if num_indexes_to_replace is None:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets !=
                                                class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(dataset=train_set, indexes=indexes_to_replace,
                        seed=seed-1, only_mark=only_mark)

    datasets = {
        "train": train_set,
        "test": test_set,
        "val": valid_set
    }
    print(len(train_set))
    for (name, ds) in datasets.items():
        writer = DatasetWriter(data_dir+f'/cifar_{name}.beton', {
            'image': RGBImageField(),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)
    loaders, _ = make_dataloaders_cifar(data_dir+'/cifar_train.beton', data_dir+'/cifar_val.beton',
                                        data_dir+'/cifar_test.beton', batch_size=batch_size, num_workers=num_workers)
    return loaders['train'], loaders['val'], loaders['test']


def cifar10_dataloaders_no_val(batch_size=128, data_dir='datasets/cifar10', num_workers=2):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = CIFAR10(data_dir, train=True,
                        transform=train_transform, download=True)
    val_set = CIFAR10(data_dir, train=False,
                      transform=test_transform, download=True)
    test_set = CIFAR10(data_dir, train=False,
                       transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(batch_size=128, data_dir='datasets/cifar100', num_workers=2):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = Subset(CIFAR100(data_dir, train=True,
                       transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform,
                     download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False,
                        transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar100_dataloaders_no_val(batch_size=128, data_dir='datasets/cifar100', num_workers=2):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-100\t 45000 images for training \t 500 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = CIFAR100(data_dir, train=True,
                         transform=train_transform, download=True)
    val_set = CIFAR100(data_dir, train=False,
                       transform=test_transform, download=True)
    test_set = CIFAR100(data_dir, train=False,
                        transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


class TinyImageNetDataset(Dataset):
    def __init__(self, image_folder_set, norm_trans=None, start=0, end=-1):
        self.imgs = []
        self.targets = []
        self.transform = image_folder_set.transform
        for sample in tqdm(image_folder_set.imgs[start: end]):
            self.targets.append(sample[1])
            img = transforms.ToTensor()(Image.open(sample[0]).convert("RGB"))
            if norm_trans is not None:
                img = norm_trans(img)
            self.imgs.append(img)
        self.imgs = torch.stack(self.imgs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.imgs[idx]), self.targets[idx]
        else:
            return self.imgs[idx], self.targets[idx]


class TinyImageNet:
    """
        TinyImageNet dataset.
    """

    def __init__(self, args, normalize=False):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ) if normalize else None

        self.tr_train = [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        self.tr_test = []

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

        self.train_path = os.path.join(args.data_dir, 'train/')
        self.val_path = os.path.join(args.data_dir, 'val/')
        self.test_path = os.path.join(args.data_dir, 'test/')

        if os.path.exists(os.path.join(self.val_path, "images")):
            if os.path.exists(self.test_path):
                os.rename(self.test_path, os.path.join(
                    args.data_dir, "test_original"))
                os.mkdir(self.test_path)
            val_dict = {}
            val_anno_path = os.path.join(self.val_path, "val_annotations.txt")
            with open(val_anno_path, 'r') as f:
                for line in f.readlines():
                    split_line = line.split('\t')
                    val_dict[split_line[0]] = split_line[1]

            paths = glob.glob(os.path.join(args.data_dir, 'val/images/*'))
            for path in paths:
                file = path.split('/')[-1]
                folder = val_dict[file]
                if not os.path.exists(self.val_path + str(folder)):
                    os.mkdir(self.val_path + str(folder))
                    os.mkdir(self.val_path + str(folder) + '/images')
                if not os.path.exists(self.test_path + str(folder)):
                    os.mkdir(self.test_path + str(folder))
                    os.mkdir(self.test_path + str(folder) + '/images')

            for path in paths:
                file = path.split('/')[-1]
                folder = val_dict[file]
                if len(glob.glob(self.val_path + str(folder) + '/images/*')) < 25:
                    dest = self.val_path + str(folder) + '/images/' + str(file)
                else:
                    dest = self.test_path + \
                        str(folder) + '/images/' + str(file)
                move(path, dest)

            os.rmdir(os.path.join(self.val_path, "images"))

    def data_loaders(self, **kwargs):

        trainset = ImageFolder(self.train_path, transform=self.tr_train)
        trainset = TinyImageNetDataset(trainset, self.norm_layer)
        testset = ImageFolder(self.test_path, transform=self.tr_test)
        testset = TinyImageNetDataset(testset, self.norm_layer)

        np.random.seed(10)

        train_loader = DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            **kwargs
        )

        np.random.seed(50)

        val_loader = DataLoader(
            testset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            **kwargs
        )

        test_loader = DataLoader(
            testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, **kwargs
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader


def cifar10_dataloaders(batch_size=128, data_dir='datasets/cifar10', num_workers=2, class_to_replace: int = None, num_indexes_to_replace=None, indexes_to_replace=None, seed: int = 1, only_mark: bool = False, shuffle=True):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Dataset information: CIFAR-10\t 45000 images for training \t 5000 images for validation\t')
    print('10000 images for testing\t no normalize applied in data_transform')
    print('Data augmentation = randomcrop(32,4) + randomhorizontalflip')

    train_set = CIFAR10(data_dir, train=True,
                        transform=train_transform, download=True)

    test_set = CIFAR10(data_dir, train=False,
                       transform=test_transform, download=True)

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    rng = np.random.RandomState(seed)
    valid_set = copy.deepcopy(train_set)
    valid_idx = []
    for i in range(max(train_set.targets) + 1):
        class_idx = np.where(train_set.targets == i)[0]
        valid_idx.append(rng.choice(class_idx, int(
            0.1*len(class_idx)), replace=False))
    valid_idx = np.hstack(valid_idx)
    train_set_copy = copy.deepcopy(train_set)

    valid_set.data = train_set_copy.data[valid_idx]
    valid_set.targets = train_set_copy.targets[valid_idx]

    train_idx = list(set(range(len(train_set)))-set(valid_idx))

    train_set.data = train_set_copy.data[train_idx]
    train_set.targets = train_set_copy.targets[train_idx]

    if class_to_replace is not None and indexes_to_replace is not None:
        raise ValueError(
            "Only one of `class_to_replace` and `indexes_to_replace` can be specified")
    if class_to_replace is not None:
        replace_class(train_set, class_to_replace, num_indexes_to_replace=num_indexes_to_replace, seed=seed-1,
                      only_mark=only_mark)
        if num_indexes_to_replace is None:
            test_set.data = test_set.data[test_set.targets != class_to_replace]
            test_set.targets = test_set.targets[test_set.targets !=
                                                class_to_replace]
    if indexes_to_replace is not None:
        replace_indexes(dataset=train_set, indexes=indexes_to_replace,
                        seed=seed-1, only_mark=only_mark)

    loader_args = {'num_workers': 0, 'pin_memory': False}

    def _init_fn(worker_id):
        np.random.seed(int(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                            worker_init_fn=_init_fn if seed is not None else None, **loader_args)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             worker_init_fn=_init_fn if seed is not None else None, **loader_args)

    return train_loader, val_loader, test_loader


def replace_indexes(dataset: torch.utils.data.Dataset, indexes, seed=0,
                    only_mark: bool = False):
    if not only_mark:
        rng = np.random.RandomState(seed)
        new_indexes = rng.choice(
            list(set(range(len(dataset))) - set(indexes)), size=len(indexes))
        dataset.data[indexes] = dataset.data[new_indexes]
        dataset.targets[indexes] = dataset.targets[new_indexes]
    else:
        # Notice the -1 to make class 0 work
        dataset.targets[indexes] = - dataset.targets[indexes] - 1


def replace_class(dataset: torch.utils.data.Dataset, class_to_replace: int, num_indexes_to_replace: int = None,
                  seed: int = 0, only_mark: bool = False):
    indexes = np.flatnonzero(np.array(dataset.targets) == class_to_replace)
    if num_indexes_to_replace is not None:
        assert num_indexes_to_replace <= len(
            indexes), f"Want to replace {num_indexes_to_replace} indexes but only {len(indexes)} samples in dataset"
        rng = np.random.RandomState(seed)
        indexes = rng.choice(
            indexes, size=num_indexes_to_replace, replace=False)
        print(f"Replacing indexes {indexes}")
    replace_indexes(dataset, indexes, seed, only_mark)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = cifar10_dataloaders()
    for i, (img, label) in enumerate(train_loader):
        print(torch.unique(label).shape)
