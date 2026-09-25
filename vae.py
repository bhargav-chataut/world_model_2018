"""
Variational Autoencoder for 64x64 RGB CarRacing frames.

The encoder compresses an image into:
    mu: mean of the latent distribution
    log_var: log variance of the latent distribution

A latent vector z is sampled from that distribution,
then the decoder reconstructs the image.
"""

import torch
import torch.nn as nn


LATENT_DIM = 32


class VAE(nn.Module):
    """
    Variational Autoencoder for 64x64 RGB images.

    Parameters:
        latent_dim: size of the latent vector
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU()
        )

        # 256 x 2 x 2 = 1024 features
        self.fc_mu = nn.Linear(
            256 * 2 * 2,
            latent_dim
        )

        self.fc_log_var = nn.Linear(
            256 * 2 * 2,
            latent_dim
        )

        # Latent vector -> 1024 decoder features
        self.decoder_input = nn.Linear(
            latent_dim,
            1024
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                1024,
                128,
                kernel_size=5,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=5,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=6,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                3,
                kernel_size=6,
                stride=2
            ),

            nn.Sigmoid()
        )


    def encode(self, x):
        """
        Encode images into latent distribution parameters.

        x: image batch with shape (N, 3, 64, 64)

        Returns:
            mu (N, latent_dim): mean of the latent distribution
            log_var (N, latent_dim): log variance of the latent distribution
        """

        x = self.encoder(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var


    def reparameterize(self, mu, log_var):
        """
        Sample latent vector z.

        Returns:
            sampled latent vector z
        """

        std = torch.exp(
            0.5 * log_var
        )

        epsilon = torch.randn_like(std)

        z = mu + std * epsilon

        return z


    def decode(self, z):
        """
        Reconstruct an image from latent vector z.
        """

        x = self.decoder_input(z)

        x = torch.relu(x)

        # 1024 values become 1024 feature maps of size 1x1
        x = x.view(
            -1,
            1024,
            1,
            1
        )

        reconstruction = self.decoder(x)

        return reconstruction


    def forward(self, x):
        """
        Run the complete VAE.

        Returns:
            reconstruction
            mu
            log_var
        """

        mu, log_var = self.encode(x)

        z = self.reparameterize(
            mu,
            log_var
        )

        reconstruction = self.decode(z)

        return reconstruction, mu, log_var