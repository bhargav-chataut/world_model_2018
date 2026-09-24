"""
Collect CarRacing-v3 episodes and save frames/actions as .npz files.

Parameters:
    num_episodes: number of episodes to collect
    output_dir: folder where episodes will be saved
    workers: number of processes collecting episodes in parallel
"""

import gymnasium as gym
import numpy as np
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


NUM_EPISODES = 10
WORKERS = 2
OUTPUT_DIR = "data"


def collect_episode(episode, output_dir):
    """
    Collect one episode.

    episode: episode number, also used as reset seed
    output_dir: directory where the .npz file is saved

    Returns:
        episode number and number of collected frames
    """

    # Each process needs its own environment.
    env = gym.make("CarRacing-v3", continuous=True)

    observation, info = env.reset(seed=episode)

    frames = []
    actions = []

    while True:
        frame = Image.fromarray(observation)
        frame = frame.resize((64, 64))
        frame = np.array(frame, dtype=np.uint8)

        frames.append(frame)

        action = env.action_space.sample()
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()

    frames = np.array(frames, dtype=np.uint8)
    actions = np.array(actions, dtype=np.float32)

    output_path = Path(output_dir) / f"episode_{episode:04d}.npz"

    np.savez(
        output_path,
        frames=frames,
        actions=actions
    )

    return episode, len(frames)


def collect_dataset(num_episodes, output_dir, workers=2):
    """
    Collect multiple episodes in parallel.

    num_episodes: total number of episodes
    output_dir: folder where episodes are saved
    workers: number of parallel processes
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ProcessPoolExecutor manages multiple worker processes.
    with ProcessPoolExecutor(max_workers=workers) as executor:

        futures = []

        for episode in range(num_episodes):
            future = executor.submit(
                collect_episode,
                episode,
                output_dir
            )

            futures.append(future)

        for future in futures:
            # result() waits for one submitted job and gets its return value.
            episode, num_frames = future.result()

            print(
                f"Episode {episode}: "
                f"{num_frames} frames"
            )


if __name__ == "__main__":
    collect_dataset(
        num_episodes=NUM_EPISODES,
        output_dir=OUTPUT_DIR,
        workers=WORKERS
    )