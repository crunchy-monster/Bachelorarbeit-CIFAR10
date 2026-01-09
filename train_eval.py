import time
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
    early_stopping=False,
    patience=15,
    monitor="test_loss",      # "test_loss" or "test_acc"
    min_delta=0.0,            # minimal improvement to count as "better"
    track_gpu_memory=True,    # records cuda peak mem if available
    verbose=True,
):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    history = {"train_loss": [], "test_loss": [], "test_acc": [], "epoch_time_sec": []}

    use_cuda = (isinstance(device, torch.device) and device.type == "cuda") or (str(device).startswith("cuda"))
    if track_gpu_memory and use_cuda:
        history["gpu_peak_mem_mb"] = []


    # Early stopping bookkeeping
    if monitor not in ("test_loss", "test_acc"):
        raise ValueError("monitor must be 'test_loss' or 'test_acc'")

    # For loss: lower is better. For acc: higher is better.
    if monitor == "test_loss":
        best_metric = float("inf")
        is_better = lambda current, best: current < (best - min_delta)
    else:
        best_metric = -float("inf")
        is_better = lambda current, best: current > (best + min_delta)

    best_epoch = -1
    best_state_dict = None
    bad_epochs = 0

    total_start = time.time()
    gpu_peak_total = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()

        if track_gpu_memory and use_cuda:
            torch.cuda.reset_peak_memory_stats()

        train_loss = train_one_epoch(model, trainloader, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, device)

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time_sec"].append(epoch_time)

        # GPU memory (peak allocated during epoch)
        if track_gpu_memory and use_cuda:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            history["gpu_peak_mem_mb"].append(peak_mb)
            gpu_peak_total = max(gpu_peak_total, peak_mb)

        # Print progress
        if verbose:
            msg = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"test_loss={test_loss:.4f} | "
                f"test_acc={test_acc:.2f}% | "
                f"time={epoch_time:.2f}s"
            )
            if track_gpu_memory and use_cuda:
                msg += f" | gpu_peak={history['gpu_peak_mem_mb'][-1]:.1f}MB"
            print(msg)

        # Early stopping check
        current_metric = test_loss if monitor == "test_loss" else test_acc
        if is_better(current_metric, best_metric):
            best_metric = current_metric
            best_epoch = epoch
            bad_epochs = 0
            # Keep best weights in memory (no file IO)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if early_stopping and bad_epochs >= patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best {monitor}={best_metric:.4f} at epoch {best_epoch + 1}."
                )
            break

    total_time = time.time() - total_start
    history["total_time_sec"] = total_time
    history["best_epoch"] = best_epoch + 1 if best_epoch >= 0 else None
    history["best_metric"] = best_metric
    history["best_state_dict"] = best_state_dict  # optional use later

    if track_gpu_memory and use_cuda:
        history["gpu_peak_mem_mb_total"] = gpu_peak_total

    return history



