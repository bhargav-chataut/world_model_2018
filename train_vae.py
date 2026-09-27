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
from torch.utils.data import Dataset, DataLoader # Gives us a way to create batches of data

from vae import VAE
from dataset import load_dataset


BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 1


class FrameDataset(Dataset):
    """
    Dataset for CarRacing frames.

    Parameters:
        frames: NumPy array of uint8 RGB frames

    Returns:
        One normalized PyTorch frame with shape (3, 64, 64)
    """

    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]

        frame = torch.from_numpy(frame).float()
        frame = frame.permute(2, 0, 1)
        frame = frame / 255.0

        return frame


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

    # KL Loss = -0.5 * sum(1 + log_var - mu^2 - var)
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

    train_dataset = FrameDataset(train_frames)
    val_dataset = FrameDataset(val_frames)

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

        # Loop through batches of training data.
        for x in train_loader:

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

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for x in val_loader:

                x = x.to(device)

                reconstruction, mu, log_var = model(x)

                loss, reconstruction_loss, kl_loss = vae_loss(
                    reconstruction,
                    x,
                    mu,
                    log_var
                )

                val_loss += loss.item()

        average_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Train Loss: {average_train_loss:.2f} "
            f"| Val Loss: {average_val_loss:.2f}"
        )
    return model