import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Directory containing saved sweep files
sweeps_dir = 'sweeps'

# Find all background and signal loss files
background_files = sorted([f for f in os.listdir(sweeps_dir) if 'background_test_loss' in f])
signal_files = sorted([f for f in os.listdir(sweeps_dir) if 'signal_loss' in f and 'background' not in f])

# Initialize loss plot
plt.figure(figsize=(10, 6))

# Plot all background and signal losses
for idx, (bg_file, sig_file) in enumerate(zip(background_files, signal_files)):
    background_loss = np.array(np.load(os.path.join(sweeps_dir, bg_file), allow_pickle=True))
    signal_loss = np.array(np.load(os.path.join(sweeps_dir, sig_file), allow_pickle=True))

    epochs = np.arange(1, 21)

    plt.plot(epochs, background_loss, linestyle='--', alpha=0.6, label=f'Run {idx} Background')
    plt.plot(epochs, signal_loss, linestyle='-', alpha=0.6, label=f'Run {idx} Signal')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('All Trials: Background vs Signal Loss Over Epochs')
plt.legend(fontsize=6, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('sweeps/all_trials_loss_curves.png')
print("✅ Loss curves saved to sweeps/all_trials_loss_curves.png")

# Initialize ROC plot
plt.figure(figsize=(8, 8))

# Plot ROC curves
for idx, (bg_file, sig_file) in enumerate(zip(background_files, signal_files)):
    background_loss = np.array(np.load(os.path.join(sweeps_dir, bg_file), allow_pickle=True))
    signal_loss = np.array(np.load(os.path.join(sweeps_dir, sig_file), allow_pickle=True))



    y_true = np.concatenate([np.zeros(len(background_loss)), np.ones(len(signal_loss))])
    y_scores = np.concatenate([background_loss, signal_loss])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'Run {idx} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Trials')
plt.legend(fontsize=6)
plt.grid(True)
plt.tight_layout()
plt.savefig('sweeps/all_trials_roc_curves.png')
print("✅ ROC curves saved to sweeps/all_trials_roc_curves.png")
