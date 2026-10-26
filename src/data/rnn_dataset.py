"""
Create Dataset for training the RNN.

Requirement: latent_dir containing encoded .npz episode files with mu, log_var, actions

Parameters:
    latent_dir: folder containing encoded episode files
    sequence_length: number of consecutive timesteps in one sample
    val_split: percentage of whole episodes used for validation

Output:
    train_dataset: RNNDataset containing training sequences
    val_dataset: RNNDataset containing validation sequences

Each sample contains:
    current_sequence:(sequence_length, 35) 32 mu values + 3 action values
    next_sequence: (sequence_length, 32) 32 next mu values

For now, log_var is stored in the encoded files but is not used
for the deterministic RNN baseline.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RNNDataset(Dataset):
    def __init__(self, latent_dir, sequence_length=32):
        """
        Load encoded episodes and create valid sequence starting points.
        """
        self.sequence_length = sequence_length

        episode_files = sorted(
            Path(latent_dir).glob("episode_*.npz")
        )

        self.episodes = []
        self.samples = []

        for file_index, file in enumerate(episode_files):
            data = np.load(file)

            mu = torch.from_numpy(data["mu"]).float()
            actions = torch.from_numpy(data["actions"]).float()

            self.episodes.append(
                (mu, actions)
            )

            episode_length = len(mu)

            number_of_sequences = (
                episode_length - sequence_length
            )

            for start in range(number_of_sequences):
                self.samples.append(
                    (file_index, start)
                )

    def __len__(self):
        """
        Return the total number of available sequence samples.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Return one sequence for RNN training.
        """
        file_index, start = self.samples[index]

        mu, actions = self.episodes[file_index]

        end = start + self.sequence_length

        current_mu = mu[start:end]
        current_actions = actions[start:end]

        next_mu = mu[start + 1:end + 1]

        inputs = torch.cat(
            [current_mu, current_actions],
            dim=-1
        )

        targets = next_mu

        return inputs, targets


if __name__ == "__main__":

    dataset = RNNDataset(
        latent_dir="latent_data",
        sequence_length=32
    )

    print("Number of sequences:", len(dataset))

    inputs, targets = dataset[0]

    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)