import os
from PIL import Image
import h5py

import torch
from torchvision import transforms, datasets
from torchvision.datasets import VisionDataset


def get_image_loader(dname, batch_size, cuda, workers, distributed, data_dir='../../data', subset=None):

    assert dname in ['MNIST', 'Fashion', 'SVHN', 'CIFAR10', 'CIFAR100', 'SmallImagenet', 'Imagenet']

    if dname == 'MNIST':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True,
                                       transform=transform_train)
        val_dataset = datasets.MNIST(root=data_dir, train=False, download=True,
                                     transform=transform_test)
        input_channels = 1
        N_classes = 10

    elif dname == 'Fashion':

        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])

        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True,
                                              transform=transform_train)
        val_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True,
                                            transform=transform_test)
        input_channels = 1
        N_classes = 10

    elif dname == 'SVHN':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform_train)

        val_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform_test)
        input_channels = 3
        N_classes = 10

    elif dname == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)

        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        input_channels = 3
        N_classes = 10

    elif dname == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)

        val_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        input_channels = 3
        N_classes = 100

    elif dname == 'Imagenet':
        traindir = os.path.join(data_dir, 'imagenet/train')
        valdir = os.path.join(data_dir, 'imagenet/val')
        imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                imagenet_normalize,
            ]))

        input_channels = 3
        N_classes = 1000

    elif dname == 'SmallImagenet':
        # h5file_path = os.path.join(data_dir, 'small_imagenet.h5')
        traindir = os.path.join(data_dir, 'imagenet84/train')
        valdir = os.path.join(data_dir, 'imagenet84/val')
        small_imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transform=transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                small_imagenet_normalize
            ])
        )
        val_dataset = datasets.ImageFolder(
            valdir, transform=transforms.Compose([
                transforms.ToTensor(),
                small_imagenet_normalize,
            ])
        )

        # train_dataset = HDF5VisionDataset(
        #     h5file_path, train=True,
        #     transform=transforms.Compose([
        #         transforms.RandomCrop(84, padding=8),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ColorJitter(
        #             brightness=0.4,
        #             contrast=0.4,
        #             saturation=0.4,
        #             hue=0.2
        #         ),
        #         transforms.ToTensor(),
        #         small_imagenet_normalize
        #     ])
        # )

        # val_dataset = HDF5VisionDataset(
        #     h5file_path, train=False,
        #     transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         small_imagenet_normalize,
        #     ])
        # )

        input_channels = 3
        N_classes = 1000

    if subset is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, subset)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=cuda, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=cuda)

    Ntrain = len(train_dataset)
    Ntest = len(val_dataset)
    print('Ntrain: %d, Nval: %d' % (Ntrain, Ntest))

    return train_sampler, train_loader, val_loader, input_channels, N_classes, Ntrain


class HDF5VisionDataset(VisionDataset):
    """HDF5 dataset for vision datasets."""

    def __init__(self, hdf5_path, train=True, transform=None, target_transform=None):
        """
        Args:
            hdf5_path (string): Path to the HDF5 file with inputs ("img") and targets ("target").
            train (bool, optional): If True, creates dataset from the "train" group,
                otherwise from the "val" group.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """
        super(HDF5VisionDataset, self).__init__(hdf5_path, transform, target_transform)

        hdf5_group = h5py.File(hdf5_path, 'r')["train" if train else "val"]
        self.img = hdf5_group["img"]
        self.targets = hdf5_group["target"]
        self.length = self.targets.len()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, target = self.img[idx], self.targets[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Val")
