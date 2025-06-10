"""
Training utility functions for JetGraphAutoencoder models.

This module includes:
- `train_loop`: One epoch of training over a data loader.
- `eval_loop`: Evaluation on a dataset with optional labeling for background/signal.
- `train_model`: Full training procedure across multiple epochs for background/signal separation.
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader

def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Executes one epoch of training.

    Args:
        dataloader (DataLoader): Dataloader for training graphs.
        model (torch.nn.Module): The autoencoder model.
        loss_fn (callable): Loss function (e.g., MSE).
        optimizer (torch.optim.Optimizer): Optimizer instance.

    Returns:
        float: Mean training loss over the epoch.
    """
    model.train()
    total_loss = []

    for batch, X in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        pred = pred[:, :X.x.shape[1]]  # Only reconstruct original feature dimensions

        loss = loss_fn(pred, X.x)
        total_loss.append(float(loss))

        loss.backward()
        optimizer.step()

    return np.nanmean(total_loss)


def eval_loop(dataloader, model, loss_fn, test=False, signal=False):
    """
    Evaluates the model on a given dataset.

    Args:
        dataloader (DataLoader): Loader containing the evaluation set.
        model (torch.nn.Module): Trained model to evaluate.
        loss_fn (callable): Loss function to use.
        test (bool): If True, stores losses as `background_test_loss`.
        signal (bool): If True, stores losses as `signal_loss`.

    Returns:
        List[float]: List of per-graph losses.
    """
    model.eval()
    loss = []
    data = []

    with torch.no_grad():
        for X in dataloader:
            pred = model(X)
            pred = pred[:, :X.x.shape[1]]
            loss.append(float(loss_fn(pred, X.x)))
            data.append(X.x)

    # Store for downstream use if flagged
    if test:
        model.test_data = data
        model.background_test_loss = loss
    elif signal:
        model.signal_data = data
        model.signal_loss = loss

    return loss


def train_model(train_dataloader, test_dataloader, signal_dataloader, model, loss_fn, optimizer, epochs, batch_size):
    """
    Trains the model across multiple epochs and tracks performance on validation and signal sets.

    Args:
        train_dataloader (DataLoader): Training data loader.
        test_dataloader (DataLoader): Background validation set.
        signal_dataloader (DataLoader): Signal evaluation set.
        model (torch.nn.Module): JetGraphAutoencoder model.
        loss_fn (callable): Loss function (e.g., MSE).
        optimizer (torch.optim.Optimizer): Optimizer instance.
        epochs (int): Number of training epochs.
        batch_size (int): Size of batches used in training.

    Returns:
        Tuple[List[float], List[float], List[float]]:
            train_loss: Mean training losses per epoch.
            val_loss: Mean validation losses per epoch.
            signal_loss: Mean signal losses per epoch.
    """
    model.train_hist = []
    model.val_hist = []
    model.signal_hist = []

    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")

        train_loss = []
        val_loss = []
        signal_loss = []

        # Run training step
        train_loop(train_dataloader, model, loss_fn, optimizer)

        # Evaluate on validation and signal
        val_loss.extend(eval_loop(test_dataloader, model, loss_fn, test=True, signal=False))
        train_loss.extend(eval_loop(train_dataloader, model, loss_fn, test=False, signal=False))
        signal_loss.extend(eval_loop(signal_dataloader, model, loss_fn, test=False, signal=True))

        # Append epoch results
        model.train_hist.append(np.nanmean(train_loss))
        model.val_hist.append(np.nanmean(val_loss))
        model.signal_hist.append(np.nanmean(signal_loss))

        print("train loss ", np.nanmean(train_loss))
        print("test loss ", np.nanmean(val_loss))
        print("signal loss", np.nanmean(signal_loss))

    # Final assignment for later ROC/score plotting
    model.background_test_loss = val_loss
    model.background_train_loss = train_loss
    model.signal_loss = signal_loss

    return train_loss, val_loss, eval_loop(signal_dataloader, model, loss_fn, test=False, signal=True)
