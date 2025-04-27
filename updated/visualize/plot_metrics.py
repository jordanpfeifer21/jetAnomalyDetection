import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd

def plot_loss(train_loss, val_loss, save_path='plots/loss.png'):
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
    plt.figure()
    bins = 100
    plt.hist(anomaly_scores, bins=bins, alpha=0.5, label=f'Signal: {signal_label}', density=True)
    plt.hist(test_scores, bins=bins, alpha=0.5, label=f'Background: {background_label}', density=True)
    plt.xlabel('Loss (MSE)')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(test_scores, signal_scores, save_path='plots/roc_curve.png'):
    fpr, tpr, _ = roc_curve(
        np.concatenate([np.zeros(len(test_scores)), np.ones(len(signal_scores))]),
        np.concatenate([test_scores, signal_scores])
    )
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    roc_data = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    roc_data.to_csv('plots/roc_data.csv', index=False)
