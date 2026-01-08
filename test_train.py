import torch
from config import BATCH_SIZE, LR, WEIGHT_DECAY
from data import get_cifar10_loaders
from models.lenet import LeNet5_CIFAR10
from train_eval import train_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader, _, _ = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        num_workers=2,   # kannst du lassen
    )

    model = LeNet5_CIFAR10().to(device)

    history = train_model(
        model,
        trainloader,
        testloader,
        device,
        epochs=10,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

if __name__ == "__main__":
    main()
