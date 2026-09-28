## Progress Report

- Day 1: Set up CarRacing-v3 data collection, collected one random episode, and saved it as a `.npz` file containing frames `(N, 64, 64, 3)` and actions `(N, 3)`.
- Day 2: Created a dataset of 100 episodes. Split them into tran, validation set. Decided to use same 2018 ConvVAE setup for now.
- Day 3: Built and understood the 2018-style ConvVAE, including CNN feature-map shapes, mu/log_var, reparameterization to sample z, and decoding back to a 64×64 RGB reconstruction.
- Day 4: The VAE is now fully trainable and we can visually inspect whether the latent representation is preserving useful information from the CarRacing frames.
- Day 5: Finished the VAE pipeline end-to-end, added checkpoint/resume support with Adam state, validated reconstructions/loss curves, and finalized the Colab notebook for GitHub.