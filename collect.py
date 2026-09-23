"""
Collect data from the CarRacing-v3 environment and save it as a numpy array.
"""


import gymnasium as gym # RL library
import numpy as np 
from PIL import Image # Image Processing library
import matplotlib.pyplot as plt # Visualization library

# Game simulator object creates a world for you
env = gym.make("CarRacing-v3", continuous=True)

# Create x0
observation, info = env.reset(seed=42)

frames = []
actions = []

i=0
while True:
    # 96x96 -> 64x64
    frame = Image.fromarray(observation)
    frame = frame.resize((64, 64))
    frame = np.array(frame, dtype=np.uint8)

    frames.append(frame)

    # Random action: [steering, gas, brake]
    action = env.action_space.sample()
    actions.append(action)

    observation, reward, terminated, truncated, info = env.step(action)
    if i% 100 == 0:
        print(f"Step: {i}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")

    if terminated or truncated:
        break
    i+=1

env.close()

frames = np.array(frames, dtype=np.uint8)
actions = np.array(actions, dtype=np.float32)

np.savez(
    "episode.npz",
    frames=frames, # (N, 64, 64, 3)
    actions=actions # (N, 3)
)

print("Frames:", frames.shape)
print("Actions:", actions.shape)
print("Saved episode.npz")
