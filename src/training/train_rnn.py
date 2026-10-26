"""
Train the deterministic RNN latent dynamics model.

Input:
    train_loader: batches of input sequences and next-latent targets
    val_loader: validation batches
    model: RNNModel

Training objective:
    Predict next latent mu using MSE loss.

Output:
    Trained RNN model and training/validation losses.
"""

import torch
import torch.nn as nn


def train_rnn(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=5,
    learning_rate=1e-3
):
    """
    Train and validate the RNN model.
    """

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(num_epochs):

        # Training
        model.train()

        total_train_loss = 0

        for inputs, targets in train_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = model(inputs)

            loss = criterion(
                predictions,
                targets
            )

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        # Validation
        model.eval()

        total_val_loss = 0

        with torch.no_grad():

            for inputs, targets in val_loader:

                inputs = inputs.to(device)
                targets = targets.to(device)

                predictions = model(inputs)

                loss = criterion(
                    predictions,
                    targets
                )

                total_val_loss += loss.item()

        average_val_loss = (
            total_val_loss / len(val_loader)
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {average_train_loss:.6f} | "
            f"Val Loss: {average_val_loss:.6f}"
        )