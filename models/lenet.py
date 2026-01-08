# LeNet architecture
# 1x32x32 Input <- (5x5), s=1, p=0 -> avg pool s=2, p=0 -> (5x5), s=1,p=0 -> avg pool s=2 P=0
# -> Conv 5x5 to 120 channels x Linear 84 x Linear 10

import torch
import torch.nn as nn

class LeNet5_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)  # 32x32 input shape
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 32/3 = 16 shape

        # Block 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)    # 16/2 = 8 shape

        # Flatten
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.Tanh()
        # Output layer (logits for CrossEntropyLoss)
        self.fc3 = nn.Linear(84, num_classes)


    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))

        x = self.flatten(x)

        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)

        return x
