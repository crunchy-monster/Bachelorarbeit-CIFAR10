import time
import copy
import torch
import torch.nn.functional as F
import json
import csv
from pathlib import Path
from fgsm import fgsm_attack
import pandas as pd
import matplotlib.pyplot as plt

from data import get_cifar10_loaders

def train_one_epoch(model, trainloader, optimizer, device):
    model.train()
    loss_sum = 0.0

    for images, labels in trainloader:
        images = images.to(device) # Daten werden auf CPU/GPU geladen
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True) # Gradient wird gelöscht
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    return loss_sum / len(trainloader)

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    early_stopping=True,
    patience=15,
    min_delta=0,               # UNIVERSAL for val_loss
    scheduler_factor=0.1,
    scheduler_patience=5,
    track_gpu_memory=True,
    verbose=True,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Reduce LR (Learning Rate) when val_loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [],
        "epoch_time_sec": [], "lr": []
    }

    use_cuda = (device.type == "cuda")
    if track_gpu_memory and use_cuda:
        history["gpu_peak_mem_mb"] = []

    # Best-Model / Early stopping bookeeping
    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = None
    bad_epochs = 0

    for epoch in range(epochs):
        start = time.time()
        if track_gpu_memory and use_cuda:
            torch.cuda.reset_peak_memory_stats()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_clean(model, val_loader, device)

        # scheduler step on val_loss
        scheduler.step(val_loss)

        epoch_time = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_sec"].append(epoch_time)
        history["lr"].append(current_lr)

        if track_gpu_memory and use_cuda:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            history["gpu_peak_mem_mb"].append(peak_mb)

        if verbose:
            msg = (f"Epoch {epoch+1}/{epochs} | "
                   f"train_loss={train_loss:.4f} | "
                   f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}% | "
                   f"lr={current_lr:.2e} | time={epoch_time:.2f}s")
            if track_gpu_memory and use_cuda:
                msg += f" | gpu_peak={peak_mb:.1f}MB"
            print(msg)

        # Early stopping on val_loss with min_delta
        improved = val_loss < (best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if early_stopping and bad_epochs >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}. "
                      f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch+1}.")
            break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    history["best_epoch"] = best_epoch + 1 if best_epoch >= 0 else None
    history["best_val_loss"] = best_val_loss
    return history



@torch.no_grad()
def eval_clean(model, test_loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y).item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / len(test_loader), 100.0 * correct / total   # (loss, acc)

def eval_fgsm(model, test_loader, device, epsilon):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        # create adversarial batch
        x_adv = fgsm_attack(model, x, y, epsilon)

        with torch.no_grad():
            logits = model(x_adv)
            loss_sum += F.cross_entropy(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / len(test_loader), 100.0 * correct / total

@torch.no_grad()
def test_final(model, test_loader, device):
    test_loss, test_acc = eval_clean(model, test_loader, device)
    print(f"[FINAL TEST] loss={test_loss:.4f} | acc={test_acc:.2f}%")
    return test_loss, test_acc

@torch.no_grad()
def test_fgsm(model, test_loader, device, epsilon):
    adv_loss, adv_acc = eval_fgsm(model, test_loader, device, epsilon)
    print(
        f"[FGSM TEST] eps={epsilon:.5f} | "
        f"loss={adv_loss:.4f} | acc={adv_acc:.2f}%"
    )
    return adv_loss, adv_acc

def save_history(history, out_dir, run_name="run"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON (kompletter dict, inkl. best_epoch etc.)
    json_path = out_dir / f"{run_name}_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # CSV (epoch-wise keys)
    csv_path = out_dir / f"{run_name}_history.csv"
    rows = []
    n = len(history["train_loss"])
    for i in range(n):
        rows.append({
            "epoch": i + 1,
            "train_loss": history["train_loss"][i],
            "val_loss": history["val_loss"][i],
            "val_acc": history["val_acc"][i],
            "lr": history["lr"][i],
            "epoch_time_sec": history["epoch_time_sec"][i],
            **({"gpu_peak_mem_mb": history["gpu_peak_mem_mb"][i]} if "gpu_peak_mem_mb" in history else {})
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return str(json_path), str(csv_path)



def load_history_csv(csv_path):
    df = pd.read_csv(csv_path)
    history = {
        "train_loss": df["train_loss"].tolist(),
        "val_loss": df["val_loss"].tolist(),
        "val_acc": df["val_acc"].tolist(),
        "lr": df["lr"].tolist(),
    }
    # optional keys
    if "train_acc" in df.columns:
        history["train_acc"] = df["train_acc"].tolist()
    return history
