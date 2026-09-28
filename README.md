# World Models (2018) — Reimplementation

A from-scratch reimplementation of Ha & Schmidhuber's **[World Models](https://arxiv.org/abs/1803.10122)** using the CarRacing environment.

The goal is to understand the full pipeline by building each component directly:

**pixels → VAE → MDN-RNN → controller**

## Current progress

- [x] Collect CarRacing trajectories
- [x] Train/validation dataset pipeline
- [x] Convolutional VAE
- [x] VAE checkpointing and reconstruction evaluation
- [ ] MDN-RNN latent dynamics model
- [ ] Controller
- [ ] Dream rollouts and evaluation

## VAE results

The VAE compresses each **64×64 RGB frame** into a **32-dimensional latent vector** and reconstructs the observation from that representation.

### Reconstructions

![VAE reconstructions](assets/vae_reconstructions.png)

### Training history

![VAE training history](assets/vae_training_history.png)

## Repository

- `collect.py` — collect CarRacing episodes
- `dataset.py` — training/validation data loading
- `vae.py` — convolutional VAE
- `train_vae.py` — VAE training and checkpointing
- `vizualize_vae.py` — reconstruction visualization
- `VAE_final.ipynb` — end-to-end VAE notebook
- `PROGRESS.md` — daily build log

## Why this project

Rather than treating the paper as a black box, this project rebuilds the system component by component to understand how compact visual representations, learned latent dynamics, and control fit together.

Next: **learn the environment dynamics with the MDN-RNN.**
