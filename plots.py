import matplotlib.pyplot as plt
import torch
from fgsm import fgsm_attack, denorm
import numpy as np
import torch.nn.functional as F
import random



@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu()

        preds = probs.argmax(dim=1)
        all_preds.append(preds)
        all_labels.append(labels)
        all_probs.append(probs)

    return (
        torch.cat(all_preds),
        torch.cat(all_labels),
        torch.cat(all_probs),
    )

def plot_confusion_matrix(preds, labels, class_names, normalize=True):
    num_classes = len(class_names)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.float32)

    # --- Absolute Confusion Matrix --
    for t, p in zip(labels, preds):
        cm[t, p] += 1

     # --- Normalize rows (true labels) ---
    if normalize:
        cm = cm / cm.sum(dim=1, keepdim=True)
        cm = torch.nan_to_num(cm)  # falls eine Klasse nie vorkommt

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title("Confusion Matrix")

    ax.xticks(range(num_classes), class_names, rotation=45, ha="right")
    ax.yticks(range(num_classes), class_names)
    ax.colorbar(im, ax=ax)
    ax.tight_layout()

    # optional: Zahlen einblenden
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j].item()
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if val > 0.5 else "black")

    ax.xlabel("Predicted")
    ax.ylabel("True")

    return fig

def plot_confidence_hist(probs, title="Confidence (max softmax)"):
    # probs: Tensor (N, num_classes)
    conf = probs.max(dim=1).values.numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(conf, bins=20, range=(0.0, 1.0))
    ax.xlabel("Confidence (max softmax")
    ax.ylabel("Count")
    ax.title(title)
    ax.grid(True)

    return fig

@torch.no_grad()
def predict_probs(model, x):
    logits = model(x)  # Probs für ein Batch berechnen
    return torch.softmax(logits, dim=1)

def show_clean_vs_fgsm(
        model,
        loader,
        device,
        classes,
        epsilons=(1/255, 2/255, 4/255, 8/255),
        n_per_eps=3,
        title="FGSM Visual Comparison"
):
    model.eval()

    # --- Get one batch ---
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    images = images[:n_per_eps]
    labels = labels[:n_per_eps]

    # --- Clean prediction ---
    with torch.no_grad():
        logits = model(images)
        probs_clean = torch.softmax(logits, dim=1)
        pred_clean = probs_clean.argmax(dim=1)
        conf_clean = probs_clean.max(dim=1).values

    # --- Prepare plotting ---
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    outer = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.25)

    # --- Loop over epsilons ---
    for idx, eps in enumerate(epsilons):
        row = idx // 2
        col = idx % 2

        inner = outer[row, col].subgridspec(2, n_per_eps, wspace=0.05, hspace=0.05)

        # FGSM images
        x_adv = fgsm_attack(model, images, labels, eps)

        with torch.no_grad():
            logits_adv = model(x_adv)
            probs_adv = torch.softmax(logits_adv, dim=1)
            pred_adv = probs_adv.argmax(dim=1)
            conf_adv = probs_adv.max(dim=1).values

        # Denormalize for plotting
        x_clean_plot = denorm(images).clamp(0, 1).cpu()
        x_adv_plot = denorm(x_adv).clamp(0, 1).cpu()

        # Subplot title (epsilon)
        ax_title = fig.add_subplot(outer[row, col])
        ax_title.set_title(f"ε = {eps:.4f}", fontsize=12)
        ax_title.axis("off")

        # --- Plot images ---
        for i in range(n_per_eps):
            # Clean
            ax = fig.add_subplot(inner[0, i])
            ax.imshow(x_clean_plot[i].permute(1, 2, 0))
            ax.axis("off")
            ax.set_title(
                f"T:{classes[labels[i]]}\n"
                f"P:{classes[pred_clean[i]]} ({conf_clean[i]:.2f})",
                fontsize=9,
            )

            # FGSM
            ax = fig.add_subplot(inner[1, i])
            ax.imshow(x_adv_plot[i].permute(1, 2, 0))
            ax.axis("off")
            ax.set_title(
                f"P:{classes[pred_adv[i]]} ({conf_adv[i]:.2f})",
                fontsize=9,
            )

    return fig


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
