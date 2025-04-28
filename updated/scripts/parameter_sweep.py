import sys
import os
import numpy as np

# Add the parent directory of `tests/` (which is `updated/`) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import itertools
import torch
import pandas as pd
import yaml
import os
from torch_geometric.loader import DataLoader
from models.autoencoder import JetGraphAutoencoder
from train.utils_training import train_loop, eval_loop
from preprocess.make_graphs import graph_data_loader


# Load configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths
train_file = config['data']['processed_data_dir'] + config['data']['train_file']
test_file = config['data']['processed_data_dir'] + config['data']['test_file']

# Load data
datatype1 = pd.read_pickle(train_file)
datatype2 = pd.read_pickle(test_file)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter grids
learning_rates = [1e-5, 5e-5, 1e-4]
weight_decays = [0.0, 1e-4, 1e-3]
nearest_neighbors = [8, 16, 32]
smallest_dims = [8, 16]
batch_size = config['model']['batch_size']
epochs = config['training']['epochs']

# Results
results = []

# Ensure sweeps directory exists
if not os.path.exists('sweeps'):
    os.makedirs('sweeps')

# Sweep all combinations
for idx, (lr, wd, k_nn, s_dim) in enumerate(itertools.product(learning_rates, weight_decays, nearest_neighbors, smallest_dims)):
    print(f"Training with lr={lr}, weight_decay={wd}, k_nn={k_nn}, smallest_dim={s_dim}")

    # Make graphs with different k-nearest neighbors
    graphs_train = graph_data_loader(datatype1, data_label=0, nearest_neighbors=k_nn)
    graphs_test = graph_data_loader(datatype2, data_label=1, nearest_neighbors=k_nn)

    train_size = int(0.8 * len(graphs_train))
    train_graphs = graphs_train[:train_size]
    val_graphs = graphs_train[train_size:]
    signal_graphs = graphs_test

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    signal_loader = DataLoader(signal_graphs, batch_size=batch_size)

    # Build model
    model = JetGraphAutoencoder(
        num_features=train_graphs[0].x.shape[1],
        smallest_dim=s_dim,
        num_reduced_edges=k_nn
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()

    model.train_hist = []
    model.val_hist = []
    model.signal_hist = []

    # Train
    best_val_loss = float('inf')
    for epoch in range(epochs):
        epoch_train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        epoch_val_loss = np.nanmean(eval_loop(val_loader, model, loss_fn))
        epoch_signal_loss = np.nanmean(eval_loop(signal_loader, model, loss_fn))

        model.train_hist.append(epoch_train_loss)
        model.val_hist.append(epoch_val_loss)
        model.signal_hist.append(epoch_signal_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
        
    model.background_test_loss = model.val_hist
    model.signal_loss = model.signal_hist


    # Store results
    run_results = {
        'learning_rate': lr,
        'weight_decay': wd,
        'nearest_neighbors': k_nn,
        'smallest_dim': s_dim,
        'train_loss_hist': model.train_hist,
        'val_loss_hist': model.val_hist,
        'signal_loss_hist': model.signal_hist,
        'best_val_loss': best_val_loss
    }
    results.append(run_results)

    # Save individual run
    run_df = pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'train_loss': model.train_hist,
        'val_loss': model.val_hist,
        'signal_loss': model.signal_hist
    })
    run_df.to_csv(f'sweeps/run_{idx}_losses.csv', index=False)

    params_df = pd.DataFrame([{key: val for key, val in run_results.items() if key in ['learning_rate', 'weight_decay', 'nearest_neighbors', 'smallest_dim']}])
    params_df.to_csv(f'sweeps/run_{idx}_params.csv', index=False)
    # Save model loss arrays for full plotting later
    np.save(f'sweeps/run_{idx}_background_test_loss.npy', np.array(model.background_test_loss))
    np.save(f'sweeps/run_{idx}_signal_loss.npy', np.array(model.signal_loss))

# Save overall results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('sweeps/autoencoder_param_sweep.csv', index=False)
print("✅ Parameter sweep finished!")