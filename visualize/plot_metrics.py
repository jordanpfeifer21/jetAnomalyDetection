"""
Visualization utilities for analyzing training and anomaly detection performance.

This module includes:
- Loss curves for training/validation
- Anomaly score distribution histograms
- ROC curve evaluation based on reconstruction loss

These plots are designed for models trained on jet physics data using autoencoders.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd


def plot_loss(train_loss, val_loss, save_path='plots/loss.png'):
    """
    Plot training and validation loss curves over epochs.

    Args:
        train_loss (List[float]): Training loss values for each epoch.
        val_loss (List[float]): Validation loss values for each epoch.
        save_path (str): Path to save the output plot.

    Returns:
        None. Saves plot to disk.
    """
    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_anomaly_score(test_scores, anomaly_scores, background_label, signal_label, save_path='plots/anomaly_score.png'):
    """
    Plot a histogram comparing the anomaly scores (MSE loss) for signal and background samples.

    Args:
        test_scores (List[float]): MSE losses for background (QCD) data.
        anomaly_scores (List[float]): MSE losses for signal events (e.g. WJets).
        background_label (str): Label to annotate the background histogram.
        signal_label (str): Label to annotate the signal histogram.
        save_path (str): File path to save the plot.

    Returns:
        None. Saves plot to disk.
    """
    plt.figure()
    bins = 100
    range_ = (
        min(np.min(anomaly_scores), np.min(test_scores)),
        max(np.max(anomaly_scores), np.max(test_scores))
    )

    plt.hist(anomaly_scores, bins=bins, range=range_, color='red', alpha=0.5,
             label=f'signal: {signal_label}', density=True)
    plt.hist(test_scores, bins=bins, range=range_, color='blue', alpha=0.5,
             label=f'background: {background_label}', density=True)

    plt.xlabel('Loss (MSE)')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(model, signal_label, background_label, savepath, examples, loss_fn, properties=[]):
    """
    Compute and plot the ROC curve from anomaly scores for background vs. signal.

    Args:
        model (nn.Module): Trained autoencoder model containing loss attributes:
                           `background_test_loss` and `signal_loss`.
        signal_label (str): Label to annotate the signal class.
        background_label (str): Label to annotate the background class.
        savepath (str): Output path to save the ROC plot.
        examples (bool): Unused placeholder for compatibility.
        loss_fn (callable): Loss function used during training (e.g., MSELoss).
        properties (list): Reserved for optional filtering or scoring dimensions (unused).

    Returns:
        None. Prints AUC score and saves plot.
    """
    test_loss = model.background_test_loss
    anomaly_loss = model.signal_loss

    min_val = np.min([np.min(anomaly_loss), np.min(test_loss)])
    max_val = np.max([np.max(anomaly_loss), np.max(test_loss)])
    thresholds = np.linspace(min_val, max_val, num=500)

    e_signal = []     # True positive rate
    e_background = [] # False positive rate

    for threshold in thresholds:
        pred_signal = (test_loss > threshold).flatten()
        pred_background = (anomaly_loss > threshold).flatten()
        true_signal = np.ones_like(pred_signal)
        true_background = np.ones_like(pred_background)

        tp_signal = np.sum(np.logical_and(pred_signal, true_signal))
        tp_background = np.sum(np.logical_and(pred_background, true_background))

        tot_signal = len(true_signal)
        tot_background = len(true_background)

        e_signal.append(tp_signal / tot_signal)
        e_background.append(tp_background / tot_background)

    # Plot ROC
    plt.figure()
    plt.plot(e_signal, e_background, label='Autoencoder')
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel(f'e_signal: {signal_label}')
    plt.ylabel(f'e_background: {background_label}')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    auc_score = auc(e_signal, e_background)
    print(f'AUC: {auc_score:.3f}')
    plt.savefig(savepath)
    plt.close()
