import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd 
import numpy as np 
from scipy.spatial.distance import cdist


background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)
data = [background]

def make_graph(pt, eta, phi): 
    num_nodes = len(pt) 

    node_pairs = np.column_stack((eta, phi))

    distances = cdist(node_pairs, node_pairs)
    edge1 = [] 
    edge2 = []
    for i in range(num_nodes): 
        closest_indices = np.argsort(distances[i][1:11]) # add edges between the 10 closest nodes 
        edge1.extend(closest_indices)
        edge2.extend(i*np.ones(closest_indices.shape))
    edge_index = torch.tensor([edge1,
                           edge2], dtype=torch.int)
    g = Data(x=torch.tensor(np.reshape(pt, (-1, 1)), dtype=torch.float64), edge_index=edge_index)
    return g 

def get_graphs(data):
    graphs = []
    for d in data: 
        for i in range(len(d['eta'].tolist())): 
            pt = d['pt'][i]
            eta = d['eta'][i]
            phi = d['phi'][i]
            graphs.append(make_graph(pt, eta, phi))
    return graphs

graphs = get_graphs(data)

class MyDataset(InMemoryDataset):
    def __init__(self, data_list, root=None, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        self.save(self.data_list, self.processed_paths[0])

dataset2 = MyDataset(graphs) # successfully saved the graphs ! probably only super useful once we know exactly which variables we want to look at
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

dataset1 = Planetoid("\..", "CiteSeer", transform=T.NormalizeFeatures())
dataset1.data

# from torch_geometric.loader import DataLoader

# data_list = graphs
# loader = DataLoader(data_list, batch_size=32)
# print(loader)
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit

for dataset in [dataset2]: 
    print("_____________")
    print(dataset[0])
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask = None
    print(data)
    transform = RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = transform(data)
    # data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)


import argparse
import os.path as osp
import time

import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv
from torch_geometric.data import Data, InMemoryDataset
from scipy.spatial.distance import cdist

import numpy as np 
import pandas as pd 

from torch_geometric.transforms import RandomLinkSplit


parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=5)
args = parser.parse_args()

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)
data = [background]

def make_graph(pt, eta, phi): 
    num_nodes = len(pt) 

    node_pairs = np.column_stack((eta, phi))

    distances = cdist(node_pairs, node_pairs)
    edge1 = [] 
    edge2 = []
    for i in range(num_nodes): 
        closest_indices = np.argsort(distances[i][1:11]) # add edges between the 10 closest nodes 
        edge1.extend(closest_indices)
        edge2.extend(i*np.ones(closest_indices.shape))
    edge_index = torch.tensor([edge1,
                           edge2], dtype=torch.long)
    g = Data(x=torch.tensor(np.reshape(pt, (-1, 1)), dtype=torch.float64), edge_index=edge_index)
    return g 

def get_graphs(data):
    graphs = []
    for d in data: 
        for i in range(len(d['eta'].tolist())): 
            pt = d['pt'][i]
            eta = d['eta'][i]
            phi = d['phi'][i]
            graphs.append(make_graph(pt, eta, phi))
    return graphs

graphs = get_graphs(data)

class MyDataset(InMemoryDataset):
    def __init__(self, data_list, root=None, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        self.save(self.data_list, self.processed_paths[0])

dataset2 = MyDataset(graphs) # successfully saved the graphs ! probably only super useful once we know exactly which variables we want to look at


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
])
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, args.dataset, transform=transform)
# dataset2 = dataset2[0]
# dataset2 = transform(dataset2)
# transform = RandomLinkSplit(is_undirected=True)
# train_data, val_data, test_data = transform(dataset2)
# train_data, val_data, test_data = dataset[0]


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


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels)
        self.conv_logstd = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


in_channels, out_channels = dataset.num_features, 16

if not args.variational and not args.linear:
    model = GAE(GCNEncoder(in_channels, out_channels))
elif not args.variational and args.linear:
    model = GAE(LinearEncoder(in_channels, out_channels))
elif args.variational and not args.linear:
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
elif args.variational and args.linear:
    model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")




#     print(data)
#     print("TRAIN ", train_data)
#     print("VAL", val_data)
#     print("TEST", test_data)

# class GCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GCNEncoder, self).__init__()
#         self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
#         self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index).relu()
#         return self.conv2(x, edge_index)
    
# from torch_geometric.nn import GAE

# # parameters
# out_channels = 2
# num_features = dataset.num_features
# epochs = 100

# # model
# model = GAE(GCNEncoder(num_features, out_channels))

# # move to GPU (if available)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# x = data.x.to(device)
# print(x)
# train_pos_edge_index = train_data.edge_label_index.to(device)

# # inizialize the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# def train():
#     model.train()
#     optimizer.zero_grad()
#     z = model.encode(x, train_pos_edge_index)
#     loss = model.recon_loss(z, train_pos_edge_index)
#     #if args.variational:
#     #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
#     loss.backward()
#     optimizer.step()
#     return float(loss)


# def test(pos_edge_index, neg_edge_index):
#     model.eval()
#     with torch.no_grad():
#         z = model.encode(x, train_pos_edge_index)
#     return model.test(z, pos_edge_index, neg_edge_index)

# for epoch in range(1, epochs + 1):
#     loss = train()

#     auc, ap = test(test_data.edge_label_index, val_data.edge_label_index)
#     print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

# print((data.test_pos_edge_index, data.test_neg_edge_index))
# print(test_data.edge_label_index, val_data.edge_label_index)

# # print(train_data.e)

# import argparse
# import os.path as osp
# import time

# import torch

# import torch_geometric.transforms as T
# from torch_geometric.datasets import Planetoid
# from torch_geometric.nn import GAE, VGAE, GCNConv

# parser = argparse.ArgumentParser()
# parser.add_argument('--variational', action='store_true')
# parser.add_argument('--linear', action='store_true')
# parser.add_argument('--dataset', type=str, default='Cora',
#                     choices=['Cora', 'CiteSeer', 'PubMed'])
# parser.add_argument('--epochs', type=int, default=400)
# args = parser.parse_args()

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
#     device = torch.device('cpu')

# transform = T.Compose([
#     T.NormalizeFeatures(),
#     T.ToDevice(device),
#     T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
#                       split_labels=True, add_negative_train_samples=False),
# ])
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid(path, args.dataset, transform=transform)
# train_data, val_data, test_data = dataset[0]


# # _________________________________________________________

# # _________________________________________________________

# # _________________________________________________________
