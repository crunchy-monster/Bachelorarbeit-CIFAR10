# 1) Load and normalize CIFAR10
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

MEAN = (0.5,0.5,0.5)
STD = (0.5, 0.5, 0.5)

# Loading the dataset and preprocessing
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.cs.toronto.edu/~kriz/cifar.html

# https://arxiv.org/pdf/1804.07612 begründet
def get_cifar10_loaders(
		batch_size: int = 8,
		root: str = "./data",
		num_workers: int = 2,
		download: bool = True,
):
	"""
	Returns: trainloader, testloader, trainset, testset
	"""

	transform = transforms.Compose([
		transforms.ToTensor(),
	    transforms.Normalize(MEAN, STD)
	])  # Werte liegen zwischen [-1,1]

	#batch_size = 8  # https://arxiv.org/pdf/1804.07612 begründet

	trainset = torchvision.datasets.CIFAR10(
	    root=root, train=True, download=True, transform=transform)

	trainloader = torch.utils.data.DataLoader(
	    trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	testset = torchvision.datasets.CIFAR10(
	    root=root, train=False, download=True, transform=transform)

	testloader = torch.utils.data.DataLoader(
	    testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

	return trainloader, testloader, trainset, testset





def show_first_images(dataset, classes, n=25):
    images = torch.stack([dataset[i][0] for i in range(n)])
    labels = torch.tensor([dataset[i][1] for i in range(n)])

    side = int(n ** 0.5)
    plt.figure(figsize=(10, 10))
    for i in range(n):
	    img = images[i] * 0.5 + 0.5
	    img = img.permute(1, 2, 0)
	    plt.subplot(5, 5, i + 1)
	    plt.xticks([])
	    plt.yticks([])
	    plt.grid(False)
	    plt.imshow(img)
	    plt.xlabel(classes[int(labels[i])])
    plt.show()
