import matplotlib.pyplot as plt
import torch
from fgsm import fgsm_attack, denorm

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