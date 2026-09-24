# world_model_2018
Reimplementation of Ha &amp; Schmidhuber’s World Models (2018), built from scratch with a focus on understanding each component and evaluating how the model learns and predicts environment dynamics.

VAE Architectures:
1) ConVAE setup (same as 2018)

64x64 RGB image
↓
4 convolution layers
↓
32-D latent distribution
↓
sample z
↓
4 transposed-convolution layers
↓
64x64 RGB reconstruction