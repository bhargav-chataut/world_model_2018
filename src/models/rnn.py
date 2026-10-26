"""
Deterministic RNN for predicting the next VAE latent state.

Input:
    (batch_size, sequence_length, 35)
    32 latent mu values + 3 action values

Output:
    (batch_size, sequence_length, 32)
    predicted next latent mu values
"""

import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(
        self,
        input_size=35,
        hidden_size=256,
        latent_dim=32
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_size,
            latent_dim
        )

    def forward(self, x):
        rnn_output, hidden = self.rnn(x)

        predicted_mu = self.output_layer(
            rnn_output
        )

        return predicted_mu