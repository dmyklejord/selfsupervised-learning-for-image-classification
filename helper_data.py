import torch
import torchvision

from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from PIL import Image

import numpy as np


# Set image augmentations. These are the default SimCLR augmentations.
class simclr_transforms():
    size = 224
    s = 1
    color_jitter = torchvision.transforms.ColorJitter(
                0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    train = torchvision.transforms.Compose([
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
    ])
    test= torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=size),
                    torchvision.transforms.ToTensor(),
    ])


class CIFAR10Pair(CIFAR10):
    """
    From: https://github.com/leftthomas/SimCLR/blob/master/utils.py
    A stochastic data augmentation module that transforms
    any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        # Transform for the knn feature bank function:
        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def __len__(self):
        return len(self.data)


def get_dataloaders_cifar10(
    train_transforms=simclr_transforms.train,
    test_transforms=simclr_transforms.test,
    batch_size=128,
    num_workers=0):

    train_simclr_dataset = CIFAR10Pair(
        root='Cifar10_data',
        download=True,
        train=True,
        transform=train_transforms)

    test_simclr_dataset = CIFAR10Pair(
        root='Cifar10_data',
        download=True,
        train=False,
        transform=test_transforms)

    train_loader = DataLoader(
        train_simclr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    test_loader = DataLoader(
        test_simclr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    return train_loader, test_loader


class ImagePair(torchvision.datasets.ImageFolder):
    """
    Returns the same image transformed two ways.
    Modified from: https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py
    """
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample1 ,sample2, target) where target is class_index of the target class.
            sample1 and sample2 are augmented/transformed versions of sample (sample==image).
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample1 = self.transform(sample)
            sample2 = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample1, sample2, target

    def __len__(self):
        return len(self.samples)

def get_dataloaders(
    dir='data',
    train_transforms=simclr_transforms.train,
    test_transforms=simclr_transforms.test,
    batch_size=128,
    test_fraction=0.20,
    num_workers=0):

    # Getting transformed image pairs of the seedling data:
    train_dataset = ImagePair(root=dir, transform=train_transforms)
    test_dataset = ImagePair(root=dir, transform=test_transforms)

    # Splitting into training and testing datasets
    # using the test_fraction split:
    length_data = len(train_dataset)
    dataset_indices = range(length_data)
    length_test = int(test_fraction * length_data)
    train_sampler, test_sampler = random_split(dataset_indices, [length_data-length_test, length_test], generator=torch.Generator().manual_seed(123))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              drop_last=True,
                              sampler=train_sampler)


    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             sampler=test_sampler)

    return train_loader, test_loader
