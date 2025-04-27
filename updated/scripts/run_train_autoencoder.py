import sys
import os

# Add the parent directory of `tests/` (which is `updated/`) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Updated train_autoencoder.py
import torch
import pandas as pd
import os
import yaml
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

# Create graphs
datatype1_graphs = graph_data_loader(datatype1, data_label=0, nearest_neighbors=config['misc']['k_nearest_neighbors'])
datatype2_graphs = graph_data_loader(datatype2, data_label=1, nearest_neighbors=config['misc']['k_nearest_neighbors'])

# Split graphs for training
train_size = int(0.8 * len(datatype1_graphs))
train_graphs = datatype1_graphs[:train_size]
test_graphs = datatype1_graphs[train_size:]
signal_graphs = datatype2_graphs

def run_autoencoder_training(train_graphs, test_graphs, signal_graphs, save_dir='checkpoints', smallest_dim=16, num_reduced_edges=16, batch_size=64, epochs=20, initial_lr=5e-6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = JetGraphAutoencoder(
        num_features=train_graphs[0].x.shape[1],
        smallest_dim=smallest_dim,
        num_reduced_edges=num_reduced_edges
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_graphs, batch_size=batch_size)
    signal_loader = DataLoader(signal_graphs, batch_size=batch_size)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_loss, val_loss, signal_loss = [], [], []

    for epoch in range(epochs):
        epoch_train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        epoch_val_loss, _ = eval_loop(val_loader, model, loss_fn)
        epoch_signal_loss, _ = eval_loop(signal_loader, model, loss_fn)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        signal_loss.append(epoch_signal_loss)

        torch.save(model.state_dict(), f"{save_dir}/autoencoder_epoch_{epoch+1}.pt")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Signal Loss: {epoch_signal_loss:.4f}")

    losses_df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'signal_loss': signal_loss
    })
    losses_df.to_csv(f"{save_dir}/autoencoder_losses.csv", index=False)

    return model, train_loss, val_loss, signal_loss

model, train_loss, val_loss, signal_loss = run_autoencoder_training(
    train_graphs,
    test_graphs,
    signal_graphs,
    smallest_dim=config['model']['smallest_dim'],
    num_reduced_edges=config['model']['num_reduced_edges'],
    batch_size=config['model']['batch_size'],
    epochs=config['training']['epochs'],
    initial_lr=config['training']['initial_lr']
)