import matplotlib.pyplot as plt
import torch
from fgsm import fgsm_attack, denorm
import numpy as np
import torch.nn.functional as F
import random


def plot_history(history):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(epochs, history["lr"], label="lr")
    plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.yscale("log")
    plt.legend(); plt.grid(True)
    plt.show()


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu()

        preds = probs.argmax(dim=1)
        all_preds.append(preds)
        all_labels.append(y)
        all_probs.append(probs)

    return (
        torch.cat(all_preds),
        torch.cat(all_labels),
        torch.cat(all_probs),
    )

def plot_confusion_matrix(preds, labels, class_names):
    num_classes = len(class_names)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(labels, preds):
        cm[t, p] += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
    plt.yticks(range(num_classes), class_names)
    plt.colorbar()
    plt.tight_layout()

    # optional: Zahlen einblenden
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_confidence_hist(probs, title="Confidence (max softmax)"):
    conf = probs.max(dim=1).values.numpy()
    plt.figure()
    plt.hist(conf, bins=20)
    plt.xlabel("Confidence");
    plt.ylabel("Count");
    plt.title(title)
    plt.grid(True)
    plt.show()

@torch.no_grad()
def predict_probs(model, x):
    logits = model(x)
    return torch.softmax(logits, dim=1)

def show_clean_vs_fgsm(model, loader, device, classes, epsilon=4 / 255, n=8):
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    x = x[:n]
    y = y[:n]

    # predictions on clean
    probs_clean = predict_probs(model, x)
    pred_clean = probs_clean.argmax(dim=1)
    conf_clean = probs_clean.max(dim=1).values

    # adversarial
    x_adv = fgsm_attack(model, x, y, epsilon)
    probs_adv = predict_probs(model, x_adv)
    pred_adv = probs_adv.argmax(dim=1)
    conf_adv = probs_adv.max(dim=1).values

    # for plotting: denormalize
    x_plot = denorm(x).clamp(0, 1).cpu()
    x_adv_plot = denorm(x_adv).clamp(0, 1).cpu()

    plt.figure(figsize=(2 * n, 4))
    for i in range(n):
        # clean
        plt.subplot(2, n, i + 1)
        plt.imshow(x_plot[i].permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"T:{classes[y[i]]}\nC:{classes[pred_clean[i]]} ({conf_clean[i]:.2f})")

        # adv
        plt.subplot(2, n, n + i + 1)
        plt.imshow(x_adv_plot[i].permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"FGSM eps={epsilon:.4f}\nP:{classes[pred_adv[i]]} ({conf_adv[i]:.2f})")

    plt.tight_layout()
    plt.show()


def plot_training_curves(history, title="Training Curves"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    if "train_acc" in history:
        axes[1].plot(history["train_acc"], label="train_acc")
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def fgsm_single_image_demo(
    model,
    test_loader,
    device,
    class_names,
    epsilons=(1/255, 2/255, 4/255, 8/255),
    idx=None,
):
    model.eval()

    # get one batch
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # choose image
    if idx is None:
        idx = random.randint(0, images.size(0) - 1)

    image = images[idx:idx+1]
    label = labels[idx:idx+1]
    true_label = class_names[label.item()]

    def predict(image):
        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return pred.item(), conf.item()

    def fgsm_attack(image, label, epsilon):
        x = image.clone().detach()
        x.requires_grad_(True)

        logits = model(x)
        loss = F.cross_entropy(logits, label)

        model.zero_grad(set_to_none=True)
        loss.backward()

        x_adv = x + epsilon * x.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv.detach()

    # plotting
    fig, axes = plt.subplots(1, 1 + len(epsilons), figsize=(3*(1+len(epsilons)), 3))

    def show(img, ax, title):
        img = img.squeeze().detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    # clean
    pred, conf = predict(image)
    show(
        image,
        axes[0],
        f"Clean\nTrue: {true_label}\nPred: {class_names[pred]} ({conf*100:.1f}%)"
    )

    # FGSM
    for i, eps in enumerate(epsilons):
        adv_img = fgsm_attack(image, label, eps)
        pred, conf = predict(adv_img)
        show(
            adv_img,
            axes[i+1],
            f"ε={eps:.4f}\nPred: {class_names[pred]} ({conf*100:.1f}%)"
        )

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt


def plot_training_curves(history, title="Training Curves"):
    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Anzahl Subplots bestimmen ---
    has_acc = "train_acc" in history or "val_acc" in history
    has_lr = "lr" in history

    n_plots = 1 + int(has_acc) + int(has_lr)

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]  # konsistente Indexierung

    plot_idx = 0

    # --- Loss ---
    axes[plot_idx].plot(epochs, history["train_loss"], label="Train Loss")
    axes[plot_idx].plot(epochs, history["val_loss"], label="Val Loss")
    axes[plot_idx].set_xlabel("Epoch")
    axes[plot_idx].set_ylabel("Loss")
    axes[plot_idx].set_title("Loss")
    axes[plot_idx].legend()
    axes[plot_idx].grid(True)
    plot_idx += 1

    # --- Accuracy (optional) ---
    if has_acc:
        if "train_acc" in history:
            axes[plot_idx].plot(epochs, history["train_acc"], label="Train Acc")
        if "val_acc" in history:
            axes[plot_idx].plot(epochs, history["val_acc"], label="Val Acc")
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("Accuracy (%)")
        axes[plot_idx].set_title("Accuracy")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
        plot_idx += 1

    # --- Learning Rate (optional) ---
    if has_lr:
        axes[plot_idx].plot(epochs, history["lr"], label="LR")
        axes[plot_idx].set_xlabel("Epoch")
        axes[plot_idx].set_ylabel("Learning Rate")
        axes[plot_idx].set_yscale("log")
        axes[plot_idx].set_title("Learning Rate")
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    return fig
