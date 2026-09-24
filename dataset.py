"""
Load CarRacing frames from saved .npz episodes.

"""

import numpy as np
from pathlib import Path


def load_dataset(data_dir, val_split=0.1):
    """
    Load all frames from all saved episodes.

    data_dir: folder containing .npz episode files
    val_split: percentage of frames reserved for validation

    Returns:
    train_frames
    val_frames
    """

    data_dir = Path(data_dir)

    episode_files = list(data_dir.glob("episode_*.npz"))

    all_frames = []

    for file in episode_files:
        data = np.load(file)

        frames = data["frames"]

        all_frames.append(frames)

    # Combine frames from every episode into one array.
    all_frames = np.concatenate(all_frames, axis=0)

    # Shuffle frame order before splitting.
    indices = np.random.permutation(len(all_frames))
    all_frames = all_frames[indices]

    split_index = int(
        len(all_frames) * (1 - val_split)
    )

    train_frames = all_frames[:split_index]
    val_frames = all_frames[split_index:]

    return train_frames, val_frames


if __name__ == "__main__":

    train_frames, val_frames = load_dataset(
        data_dir="data"
    )

    print("Train frames:", train_frames.shape)
    print("Validation frames:", val_frames.shape)