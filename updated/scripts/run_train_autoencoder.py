import sys
import os

# Add the parent directory of `tests/` (which is `updated/`) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import yaml
from torch_geometric.loader import DataLoader
from models.autoencoder import JetGraphAutoencoder
from train.utils_training import train_loop, eval_loop, train_model
from preprocess.make_graphs import graph_data_loader
from visualize.plot_metrics import plot_loss, plot_anomaly_score, plot_roc_curve
import matplotlib.pyplot as plt
import os


# Load configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_file = config['data']['processed_data_dir'] + config['data']['train_file']
test_file = config['data']['processed_data_dir'] + config['data']['test_file']

datatype1 = pd.read_pickle(train_file)
datatype2 = pd.read_pickle(test_file)

datatype1_graphs = graph_data_loader(datatype1, data_label=0, nearest_neighbors=config['misc']['k_nearest_neighbors'])
datatype2_graphs = graph_data_loader(datatype2, data_label=1, nearest_neighbors=config['misc']['k_nearest_neighbors'])

train_size = int(0.8 * len(datatype1_graphs))
train_graphs = datatype1_graphs[:train_size]
test_graphs = datatype1_graphs[train_size:]
signal_graphs = datatype2_graphs

save_dir = './plots/'
os.makedirs(save_dir, exist_ok=True)

def run_autoencoder_training(train_graphs, test_graphs, signal_graphs, smallest_dim, num_reduced_edges, batch_size, epochs, initial_lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = JetGraphAutoencoder(
        num_features=train_graphs[0].x.shape[1],
        smallest_dim=smallest_dim,
        num_reduced_edges=num_reduced_edges
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    signal_loader = DataLoader(signal_graphs, batch_size=1, shuffle=False)
    train_hist = []
    val_hist = []
    train_loss, val_loss, signal_loss = train_model(train_loader, test_loader, signal_loader, model, loss_fn, optimizer, epochs=epochs, batch_size=batch_size)
    # for epoch in range(epochs):
    #     epoch_train_loss = train_loop(train_loader, model, loss_fn, optimizer)
    #     epoch_val_loss = eval_loop(test_loader, model, loss_fn)
    #     train_hist.append(epoch_train_loss)
    #     val_hist.append(epoch_val_loss)
    #     print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f}")

    # Calculate per-sample losses

    # _ = eval_loop(test_loader, model, torch.nn.MSELoss(), test=True, signal=False)
    # _ = eval_loop(signal_loader, model, torch.nn.MSELoss(), test=False, signal=True)

    plot_anomaly_score(model.signal_loss, model.background_test_loss, background_label="", signal_label="")
    plot_roc_curve(model, "signal", "background", savepath=save_dir + '/roc.png', examples=False, loss_fn=torch.nn.MSELoss(reduction='mean'))
    plot_loss(model.train_hist, model.val_hist, save_path='plots/loss.png')
    return model

run_autoencoder_training(
    train_graphs, test_graphs, signal_graphs,
    smallest_dim=config['model']['smallest_dim'],
    num_reduced_edges=config['model']['num_reduced_edges'],
    batch_size=config['model']['batch_size'],
    epochs=config['training']['epochs'],
    initial_lr=config['training']['initial_lr']
)
