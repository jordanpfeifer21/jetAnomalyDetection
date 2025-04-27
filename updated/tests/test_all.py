import sys
import os

# Add the parent directory of `tests/` (which is `updated/`) to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import pandas as pd
import yaml

# Import project modules
from models.autoencoder import JetGraphAutoencoder
from models.classifier import JetGraphAutoencoderClassification
from preprocess.make_graphs import graph_data_loader
from train.train_autoencoder import run_autoencoder_training
from train.train_classifier import run_classifier_training
from visualize.plot_metrics import plot_loss, plot_anomaly_score, plot_roc_curve

print("✅ Modules imported successfully.")

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load small data
train_file = config['data']['processed_data_dir'] + config['data']['train_file']
test_file = config['data']['processed_data_dir'] + config['data']['test_file']

print(f"Loading training data from {train_file}...")
df_train = pd.read_pickle(train_file)
df_test = pd.read_pickle(test_file)

# Build graphs
graphs_train = graph_data_loader(df_train, data_label=0, nearest_neighbors=config['misc']['k_nearest_neighbors'])
graphs_test = graph_data_loader(df_test, data_label=1, nearest_neighbors=config['misc']['k_nearest_neighbors'])

print(f"✅ Built {len(graphs_train)} training graphs and {len(graphs_test)} testing graphs.")

# ---- Test Autoencoder training ----
print("\n🚀 Testing Autoencoder Training...")

train_size = int(0.8 * len(graphs_train))
train_graphs = graphs_train[:train_size]
test_graphs = graphs_train[train_size:]
signal_graphs = graphs_test

model_ae, train_loss_ae, val_loss_ae, signal_loss_ae = run_autoencoder_training(
    train_graphs,
    test_graphs,
    signal_graphs,
    smallest_dim=config['model']['smallest_dim'],
    num_reduced_edges=config['model']['num_reduced_edges'],
    batch_size=config['model']['batch_size'],
    epochs=3,  # Quick test: only 3 epochs
    initial_lr=config['training']['initial_lr']
)

print("✅ Autoencoder training test complete!")

# ---- Test Classifier training ----
print("\n🚀 Testing Classifier Training...")

all_graphs = graphs_train + graphs_test

model_classif, losses_classif, accuracies_classif = run_classifier_training(
    all_graphs,
    smallest_dim=config['model']['smallest_dim'],
    num_reduced_edges=config['model']['num_reduced_edges'],
    batch_size=config['model']['batch_size'],
    epochs=3,  # Quick test: only 3 epochs
    initial_lr=config['training']['initial_lr']
)

print("✅ Classifier training test complete!")

# ---- Test Visualizations ----
print("\n📊 Testing Plots...")
plot_loss(train_loss_ae, val_loss_ae)
plot_anomaly_score(model_ae.background_test_loss, model_ae.signal_loss, background_label='QCD', signal_label='WJets')
plot_roc_curve(model_ae.background_test_loss, model_ae.signal_loss, background_label='QCD', signal_label='WJets')

print("\n🎉 All major modules tested successfully!")