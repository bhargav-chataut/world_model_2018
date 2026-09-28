"""
Vizualize VAE reconstructions on validation frames.
"""

import torch
import matplotlib.pyplot as plt

from vae import VAE


def show_reconstructions(model, val_loader, device, num_images=6):
    """
    Show original validation frames and VAE reconstructions.

    Parameters:
        model: trained VAE
        val_loader: validation DataLoader
        device: cpu or cuda device
        num_images: number of images to display
    """

    if num_images < 1:
        raise ValueError("num_images must be at least 1")

    model.eval()

    with torch.no_grad():
        x = next(iter(val_loader))[:num_images]
        num_images = len(x)
        x = x.to(device)

        reconstruction, mu, log_var = model(x)

    x = x.cpu()
    reconstruction = reconstruction.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(12, 5), squeeze=False)

    for i in range(num_images):
        axes[0, i].imshow(x[i].permute(1, 2, 0))
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstruction[i].permute(1, 2, 0))
        axes[1, i].axis("off")

    axes[0, (num_images - 1) // 2].set_title("Original", fontsize=14)
    axes[1, (num_images - 1) // 2].set_title("Reconstruction", fontsize=14)

    plt.subplots_adjust(hspace=0.35)
    plt.show()


def show_checkpoint_reconstructions(checkpoint_path, val_loader, device=None, num_images=6):
    """
    Load a saved VAE and show reconstructions without training.

    checkpoint_path: path to a checkpoint saved by train_vae.train
    val_loader: validation DataLoader, as used by show_reconstructions
    device: optional cpu or cuda device; selected automatically if omitted
    num_images: number of images to display
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model = VAE(latent_dim=checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    show_reconstructions(model, val_loader, device, num_images)
