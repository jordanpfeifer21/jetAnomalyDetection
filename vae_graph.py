import argparse
import os.path as osp
import time

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.data import Data, InMemoryDataset
from scipy.spatial.distance import cdist
from torch import tensor 
from torch_geometric.loader import DataLoader

import numpy as np 
import pandas as pd 

from torch_geometric.transforms import RandomLinkSplit

from utils import graph_data_loader, JetDataset

parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=50)
args = parser.parse_args()

# background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
# background = pd.read_pickle(background_file)
# print(background.shape)
signal = pd.read_pickle(signal_file)
print(signal.shape)
# data = [background]

graphs = graph_data_loader([signal.iloc[:300]])

print(graphs[0])
dataset = JetDataset(graphs) # successfully saved the graphs ! probably only super useful once we know exactly which variables we want to look at
train_dataset = test_dataset = dataset

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

in_channels, out_channels = dataset.num_features, 16

if not args.variational and not args.linear:
    model = GAE(GCNEncoder(in_channels, out_channels))
elif args.variational and not args.linear:
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        print(data)
        z = model.encode(data.x.float(), (data.edge_index))
        loss = model.recon_loss(z, data.pos_edge_label_index)
        if args.variational:
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         z = model.encode(data.x.float(), data.edge_index)
         model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
         print("asdjkfls")
     return   # Derive ratio of correct predictions.


times = []
losses = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    losses.append(loss)
    auc, ap = test(test_loader)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


