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

    plt.figure(figsize=(12, 4))

    plt.figtext(0.5, 0.95, "Original", ha="center", fontsize=12)
    plt.figtext(0.5, 0.48, "Reconstruction", ha="center", fontsize=12)

    for i in range(num_images):

        plt.subplot(2, num_images, i + 1)
        plt.imshow(x[i].permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstruction[i].permute(1, 2, 0))
        plt.axis("off")

    plt.tight_layout()
    plt.show()