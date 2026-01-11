import torch
import torch.nn.functional as F

CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1,3,1,1)
CIFAR10_STD  = torch.tensor((0.2470, 0.2435, 0.2616)).view(1,3,1,1)

def denorm(x, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    # x: normalized tensor (N,3,32,32)
    return x * std.to(x.device) + mean.to(x.device)

def norm(x, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    return (x - mean.to(x.device)) / std.to(x.device)

def fgsm_attack(model, x, y, epsilon):
    """
    x: normalized input batch (N,C,H,W)
    y: labels (N,)
    epsilon: step size in normalized space (typical CIFAR10 start: 1/255, 2/255, 4/255, 8/255)
    """
    model.eval()

    x_adv = x.detach().clone()
    x_adv.requires_grad_(True)

    # Loss berechnen
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)

    # Gradienten resetten und backward
    model.zero_grad(set_to_none=True)
    if x_adv.grad is not None:
        x_adv.grad.zero_()
    loss.backward()

    # FGSM perturbation in normalized space
    grad_sign = x_adv.grad.detach().sign()
    x_adv = x_adv + epsilon * grad_sign

    # Clamp in pixel space to keep valid image range
    x_adv_pixel = denorm(x_adv)
    x_adv_pixel = torch.clamp(x_adv_pixel, 0.0, 1.0)
    x_adv = norm(x_adv_pixel)

    return x_adv.detach()
