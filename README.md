# world_model_2018
Reimplementation of Ha &amp; Schmidhuber’s World Models (2018), built from scratch with a focus on understanding each component and evaluating how the model learns and predicts environment dynamics.

VAE Architectures:
1) ConVAE setup (similar to 2018 paper)

Input: (N, 3, 64, 64)

ENCODER

Conv2d(3 → 32, k=4, s=2) → (N, 32, 31, 31) → ReLU
Conv2d(32 → 64, k=4, s=2) → (N, 64, 14, 14) → ReLU
Conv2d(64 → 128, k=4, s=2) → (N, 128, 6, 6) → ReLU
Conv2d(128 → 256, k=4, s=2) → (N, 256, 2, 2) → ReLU
Flatten → (N, 1024)
Linear(1024 → 32) → mu (N, 32)
Linear(1024 → 32) → log_var (N, 32)
Reparameterize: z = mu + std * epsilon → (N, 32) 

DECODER

Linear(32 → 1024) → (N, 1024)
Reshape → (N, 1024, 1, 1)
ConvTranspose2d(1024 → 128, k=5, s=2) → (N, 128, 5, 5) → ReLU
ConvTranspose2d(128 → 64, k=5, s=2) → (N, 64, 13, 13) → ReLU
ConvTranspose2d(64 → 32, k=6, s=2) → (N, 32, 30, 30) → ReLU
ConvTranspose2d(32 → 3, k=6, s=2) → (N, 3, 64, 64)
Sigmoid → reconstructed image (N, 3, 64, 64)
