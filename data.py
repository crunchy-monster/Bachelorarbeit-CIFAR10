# 1) Load and normalize CIFAR10
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np

from fgsm import denorm, fgsm_attack

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Loading the dataset and preprocessing
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.cs.toronto.edu/~kriz/cifar.html

# https://arxiv.org/pdf/1804.07612 begründet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_cifar10_loaders(
    root,
    batch_size=32,
    num_workers=2,
    val_ratio=0.1,
    seed=42,
    pin_memory=True,
):
    # Train augmentation
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # No augmentation for val/test
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test_set   = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)

    # Split train into train/val
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=g)

    # IMPORTANT: val must NOT use augmentation
    # random_split keeps transform from full_train, so we override it:
    val_set.dataset.transform = test_tf

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader



def show_first_images(dataset, classes, n=25, title="CIFAR10 - First Images", mean=CIFAR10_MEAN, std=CIFAR10_STD):
    # dataset[i] -> (C,H,W) tensor, label
    imgs = torch.stack([dataset[i][0] for i in range(n)])
    labels = [dataset[i][1] for i in range(n)]

    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t  = torch.tensor(std).view(1, 3, 1, 1)

    # denormalize
    imgs = imgs * std_t + mean_t
    imgs = imgs.clamp(0, 1)

    side = int(n ** 0.5)
    fig = plt.figure(figsize=(10, 10))

    for i in range(n):
        img = imgs[i].permute(1, 2, 0)  # HWC
        plt.subplot(side, side, i + 1)
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.imshow(img)
        plt.xlabel(classes[int(labels[i])])

    plt.suptitle(title)
    #plt.show()
    return fig


