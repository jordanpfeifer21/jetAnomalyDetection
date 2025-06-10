import sys
import os

# Add the parent directory of `tests/` (which is `updated/`) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from torch_geometric.loader import DataLoader
from models.autoencoder import JetGraphAutoencoder
from preprocess.make_graphs import graph_data_loader
from scipy.stats import norm
import pandas as pd

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data
train_file = config['data']['processed_data_dir'] + config['data']['train_file']
test_file = config['data']['processed_data_dir'] + config['data']['test_file']

datatype1 = pd.read_pickle(train_file)
datatype2 = pd.read_pickle(test_file)

run_params = pd.read_csv('sweeps/run_0_params.csv')

# Create graphs
train_graphs = graph_data_loader(datatype1, data_label=0, nearest_neighbors=run_params['nearest_neighbors'])
test_graphs = graph_data_loader(datatype2, data_label=1, nearest_neighbors=run_params['nearest_neighbors'])

# --- Combine background graphs ---
mixed_background_graphs = train_graphs + test_graphs  # Or select only part if you want
signal_graphs = test_graphs  # assuming signal comes from datatype2

# --- Load model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = JetGraphAutoencoder(
    num_features=config['model']['num_features'],
    smallest_dim=run_params['smallest_dim'],
    num_reduced_edges=config['model']['num_reduced_edges']
).to(device)

# Set path to the best checkpoint
checkpoint_path = "sweeps/run_0_model_weights.pth"  # CHANGE THIS to your saved weight
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Evaluate anomaly scores ---
def compute_losses(graphs, model):
    model.eval()
    losses = []
    loss_fn = torch.nn.MSELoss(reduction='none')
    with torch.no_grad():
        for data in graphs:
            data = data.to(device)
            output = model(data)
            loss = loss_fn(output[:, :data.x.shape[1]], data.x)
            event_loss = loss.mean().item()
            losses.append(event_loss)
    return np.array(losses)

background_losses = compute_losses(mixed_background_graphs, model)
signal_losses = compute_losses(signal_graphs, model)

# --- Plot and calculate true p-values ---
def plot_bump_pvalues(background_scores, signal_scores, bins=100):
    """
    Plots p-values from a simple bump hunt between background and signal.
    Prints minimum p-value and statistical significance according to HEP standards.
    """
    combined_scores = np.concatenate([background_scores, signal_scores])

    # Histogram anomaly scores
    hist_bg, bin_edges = np.histogram(background_scores, bins=bins)
    hist_sig, _ = np.histogram(signal_scores, bins=bin_edges)

    # Background median estimation
    background_expectation = np.median(hist_bg)
    residuals = hist_sig - background_expectation

    # Avoid divide-by-zero
    errors = np.sqrt(np.maximum(background_expectation, 1.0))
    z_scores = residuals / errors
    p_values = norm.sf(z_scores)  # one-sided p-value

    # Find minimum p-value and print
    min_p_value = np.min(p_values)
    best_bin_center = (bin_edges[:-1] + bin_edges[1:]) / 2
    min_bin = best_bin_center[np.argmin(p_values)]

    print(f"Minimum p-value: {min_p_value:.4e} at anomaly score {min_bin:.4f}")

    # HEP significance levels
    alpha_evidence = 0.00135  # ~3σ
    alpha_discovery = 2.87e-7  # ~5σ

    if min_p_value < alpha_discovery:
        print(f"Result: Statistically significant at discovery level (5σ)!")
    elif min_p_value < alpha_evidence:
        print(f"Result: Statistically significant at evidence level (3σ)!")
    else:
        print(f"Result: Not statistically significant (below 3σ).")

    # Plot
    plt.figure(figsize=(10, 6))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(centers, p_values, marker='o', linestyle='-')
    plt.yscale('log')
    plt.axhline(alpha_evidence, color='orange', linestyle='--', label='3σ Evidence Threshold')
    plt.axhline(alpha_discovery, color='red', linestyle='--', label='5σ Discovery Threshold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('p-value')
    plt.title('Bump Hunt p-values (Signal Excess)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.clf()

# Run the bump hunt
plot_bump_pvalues(background_losses, signal_losses)
