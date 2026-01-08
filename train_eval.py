import torch
import torch.nn.functional as F

from config import BATCH_SIZE, LR, EPOCHS, WEIGHT_DECAY, SEED
from data import get_cifar10_loaders

def train_one_epoch(model, trainloader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in trainloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(trainloader)



@torch.no_grad()
def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss_sum += loss.item()

        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    acc = 100.0 * correct / total
    return loss_sum / len(testloader), acc




def train_model(
    model,
    trainloader,
    testloader,
    device,
    epochs,
    lr,
    weight_decay=0.0,
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history = {"train_loss": [], "test_loss": [], "test_acc": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, trainloader, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, device)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"test_acc={test_acc:.2f}%"
        )

    return history

