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

    # Use mean squared error for reconstruction loss.
    reconstruction_loss = F.mse_loss(
        reconstruction,
        x,
        reduction="sum"
    )

    # KL Loss = -0.5 * sum(1 + log_var - mu^2 - var^2)
    kl_loss = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    )

    batch_size = x.size(0)

    reconstruction_loss = reconstruction_loss / batch_size
    kl_loss = kl_loss / batch_size

    total_loss = reconstruction_loss + kl_loss

    return total_loss, reconstruction_loss, kl_loss



def train(data_dir):
    """
    Load the dataset, create batches, train the VAE, and save the model.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_frames, val_frames = load_dataset(data_dir)

    train_frames = prepare_frames(train_frames)
    val_frames = prepare_frames(val_frames)

    train_dataset = TensorDataset(train_frames)
    val_dataset = TensorDataset(val_frames)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = VAE().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        # Put the model in training mode. This is important for layers like dropout and batchnorm.
        model.train()

        train_loss = 0

        # Loop through batches of training data. (x,) is a tuple, so we unpack it to get the actual batch of images.
        for (x,) in train_loader:

            x = x.to(device)

            optimizer.zero_grad()

            reconstruction, mu, log_var = model(x)

            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                x,
                mu,
                log_var
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        average_train_loss = train_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Train Loss: {average_train_loss:.2f}"
        )

if __name__ == "__main__":
    train()