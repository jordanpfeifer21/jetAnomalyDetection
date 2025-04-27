import pandas as pd
from preprocess.make_graphs import graph_data_loader
from train.train_autoencoder import run_autoencoder_training
import yaml

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

# Run training
train_size = int(0.8 * len(datatype1_graphs))
train_graphs = datatype1_graphs[:train_size]
test_graphs = datatype1_graphs[train_size:]
signal_graphs = datatype2_graphs

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
