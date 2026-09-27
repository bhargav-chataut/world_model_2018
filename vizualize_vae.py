"""
Visualize VAE reconstructions on validation frames.
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

    for i in range(num_images):

        plt.subplot(2, num_images, i + 1)
        plt.imshow(x[i].permute(1, 2, 0))
        plt.axis("off")

        if i == 0:
            plt.ylabel("Original", fontsize=12)

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstruction[i].permute(1, 2, 0))
        plt.axis("off")

        if i == 0:
            plt.ylabel("Reconstruction", fontsize=12)

    plt.tight_layout()
    plt.show()