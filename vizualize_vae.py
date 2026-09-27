"""
Vizualize VAE reconstructions on validation frames.
"""

import torch
import matplotlib.pyplot as plt


def show_reconstructions(model, val_loader, device, num_images=6):
    """
    Show original validation frames and VAE reconstructions.

    Parameters:
        model: trained VAE
        val_loader: validation DataLoader
        device: cpu or cuda device
        num_images: number of images to display
    """

    model.eval()

    with torch.no_grad():
        x = next(iter(val_loader))
        x = x.to(device)

        reconstruction, mu, log_var = model(x)

    x = x.cpu()
    reconstruction = reconstruction.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(12, 5))

    for i in range(num_images):
        axes[0, i].imshow(x[i].permute(1, 2, 0))
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstruction[i].permute(1, 2, 0))
        axes[1, i].axis("off")

    axes[0, num_images // 2 - 2].set_title("Original", fontsize=14)
    axes[1, num_images // 2 - 2].set_title("Reconstruction", fontsize=14)

    plt.subplots_adjust(hspace=0.35)
    plt.show()