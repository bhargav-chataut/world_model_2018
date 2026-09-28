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

from pathlib import Path

import torch
import torch.nn.functional as F # Gives us loss functoins like MSELoss and KLDivLoss
from torch.utils.data import Dataset, DataLoader # Gives us a way to create batches of data

from vae import VAE
from dataset import load_dataset


BATCH_SIZE = 64
LEARNING_RATE = 1e-4


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


def create_dataloaders(data_dir):
    """Load frames and return training and validation DataLoaders."""

    train_frames, val_frames = load_dataset(data_dir)

    train_loader = DataLoader(
        FrameDataset(train_frames),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        FrameDataset(val_frames),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader


def load_checkpoint(checkpoint_path, device):
    """Restore the VAE, Adam optimizer, and number of completed epochs."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model = VAE(latent_dim=checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]
    return model, optimizer, start_epoch


def train_one_epoch(model, train_loader, optimizer, device):
    """Update model weights for one epoch and return average batch loss."""

    model.train()
    train_loss = 0.0

    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()

        reconstruction, mu, log_var = model(x)
        loss, _, _ = vae_loss(reconstruction, x, mu, log_var)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate(model, val_loader, device):
    """Evaluate without updating weights and return average batch loss."""

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            reconstruction, mu, log_var = model(x)
            loss, _, _ = vae_loss(reconstruction, x, mu, log_var)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path):
    """Save model and optimizer state; epoch is the completed epoch number."""

    torch.save(
        {
            "epoch": epoch,
            "latent_dim": model.latent_dim,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        checkpoint_path,
    )


def train(data_dir, save_dir="checkpoints", epochs=1, checkpoint_path=None):
    """
    Train the VAE and save a checkpoint after each epoch.

    save_dir: checkpoint folder; pass a mounted Drive path in Colab.
    epochs: final epoch to reach, including epochs already completed.
    checkpoint_path: optional checkpoint to resume model and optimizer state.
    Use a different folder for each run to keep earlier checkpoints.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader = create_dataloaders(data_dir)

    if checkpoint_path is not None:
        model, optimizer, start_epoch = load_checkpoint(checkpoint_path, device)
        print("Loaded checkpoint at epoch:", start_epoch)
    else:
        model = VAE().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Train Loss: {train_loss:.2f} "
            f"| Val Loss: {val_loss:.2f}"
        )
        save_path = save_dir / f"vae_epoch_{epoch + 1}.pt"
        save_checkpoint(model, optimizer, epoch + 1, train_loss, val_loss, save_path)
        print("Saved checkpoint:", save_path)

    return model, val_loader, device
