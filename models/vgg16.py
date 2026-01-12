import torch
import torch.nn as nn

class VGG16_CIFAR10(nn.Module):
    """
    VGG-16 style (13 conv layers) adapted for CIFAR-10 (3x32x32).
    ReLU + MaxPool, small CIFAR head (AdaptiveAvgPool -> Linear), logits output.
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()

        def conv_block(in_ch, out_ch, n_convs):
            layers = []
            for i in range(n_convs):
                layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3,   64, 2),   # 32 -> 16
            conv_block(64, 128, 2),   # 16 -> 8
            conv_block(128, 256, 3),  # 8  -> 4
            conv_block(256, 512, 3),  # 4  -> 2
            conv_block(512, 512, 3),  # 2  -> 1
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        head = [nn.Flatten(), nn.Linear(512, 512), nn.ReLU(inplace=True)]
        if dropout and dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(512, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
