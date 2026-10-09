"""
Encode CarRacing episode frames with the trained VAE.

Each encoded episode stores:
- mu
- log_var
- actions
"""

from pathlib import Path

import numpy as np
import torch

from vae import VAE


def load_vae(checkpoint_path, device):
    """
    Load a trained VAE checkpoint.

    Parameters:
        checkpoint_path: path to VAE checkpoint
        device: torch device

    Returns:
        Trained VAE model.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = VAE(
        latent_dim=checkpoint["latent_dim"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def encode_frames(model, frames, device, batch_size=256):
    """
    Encode frames into VAE mu and log_var values.

    Parameters:
        model: trained VAE
        frames: uint8 RGB frames
        device: torch device
        batch_size: number of frames encoded at once

    Returns:
        mu and log_var numpy arrays.
    """
    all_mu = []
    all_log_var = []

    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]

        batch = torch.from_numpy(batch).float()
        batch = batch.permute(0, 3, 1, 2)
        batch = batch / 255.0
        batch = batch.to(device)

        with torch.no_grad():
            mu, log_var = model.encode(batch)

        all_mu.append(mu.cpu())
        all_log_var.append(log_var.cpu())

    mu = torch.cat(all_mu, dim=0).numpy()
    log_var = torch.cat(all_log_var, dim=0).numpy()

    return mu, log_var


def encode_dataset(
    data_dir,
    checkpoint_path,
    output_dir,
):
    """
    Encode every CarRacing episode using the trained VAE.

    Parameters:
        data_dir: directory containing original episode files
        checkpoint_path: trained VAE checkpoint
        output_dir: directory for encoded episode files
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = load_vae(
        checkpoint_path,
        device,
    )

    episode_files = sorted(
        data_dir.glob("episode_*.npz")
    )

    print("Device:", device)
    print("Episodes:", len(episode_files))

    for file in episode_files:
        data = np.load(file)

        frames = data["frames"]
        actions = data["actions"]

        mu, log_var = encode_frames(
            model,
            frames,
            device,
        )

        output_path = output_dir / file.name

        np.savez_compressed(
            output_path,
            mu=mu,
            log_var=log_var,
            actions=actions,
        )

        print(
            file.name,
            "mu:", mu.shape,
            "log_var:", log_var.shape,
            "actions:", actions.shape,
        )