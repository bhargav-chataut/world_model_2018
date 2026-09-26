"""
Train the ConvVAE on CarRacing frames.

Steps:
    1. Load training/validation frames
    2. Convert frames to PyTorch tensors
    3. Create batches
    4. Run VAE forward pass
    5. Calculate reconstruction + KL loss
    6. Backpropagate and update weights
"""

import torch
import torch.nn.functional as F # Gives us loss functoins like MSELoss and KLDivLoss
from torch.utils.data import TensorDataset, DataLoader # Gives us a way to create batches of data

from vae import VAE
from dataset import load_dataset


DATA_DIR = "data"

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 1


def prepare_frames(frames):
    """
    Convert NumPy frames from:
        (N, 64, 64, 3)

    to PyTorch tensors:
        (N, 3, 64, 64)

    Also normalize pixel values from 0-255 to 0-1.
    """

    frames = torch.from_numpy(frames).float()

    frames = frames.permute(0, 3, 1, 2)

    frames = frames / 255.0

    return frames


def vae_loss(reconstruction, x, mu, log_var):
    """
    Calculate reconstruction loss + KL loss.

    reconstruction: VAE output image
    x: original image
    mu: latent means
    log_var: latent log variances

    Returns:
        total_loss
        reconstruction_loss
        kl_loss
    """

    reconstruction_loss = F.mse_loss(
        reconstruction,
        x,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    )

    batch_size = x.size(0)

    reconstruction_loss = reconstruction_loss / batch_size
    kl_loss = kl_loss / batch_size

    total_loss = reconstruction_loss + kl_loss

    return total_loss, reconstruction_loss, kl_loss