import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

data = np.load("episode.npz")

frames = data["frames"]
actions = data["actions"]

print("Frames shape:", frames.shape)
print("Actions shape:", actions.shape)

# Show one frame
plt.imshow(frames[200])
plt.axis("off")
plt.show()

# Save the whole episode as a GIF
imageio.mimsave(
    "ep.gif",
    frames,
    fps=30
)

print("Saved ep.gif")