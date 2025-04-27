import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import EdgeConv, TopKPooling, global_max_pool
from torch_cluster import knn_graph


class JetGraphAutoencoder(nn.Module):
    """
    Graph-based Autoencoder for jet anomaly detection.
    """
    def __init__(self, num_features: int = 3, smallest_dim: int = 8, topk=None, num_reduced_edges: int = 8):
        super(JetGraphAutoencoder, self).__init__()
        self.num_features = num_features
        self.smallest_dim = smallest_dim
        self.topk = topk
        self.num_reduced_edges = num_reduced_edges

        # Encoder layers
        self.conv1 = EdgeConv(Sequential(
            Linear(2 * self.num_features, self.smallest_dim * 2),
            ReLU(),
            Linear(self.smallest_dim * 2, self.smallest_dim * 2)
        ), aggr='max')

        self.conv2 = EdgeConv(Sequential(
            Linear(self.smallest_dim * 2 * 2, self.smallest_dim),
            ReLU(),
            Linear(self.smallest_dim, self.smallest_dim * 2)
        ), aggr='max')

        # Decoder layers
        self.conv3 = EdgeConv(Sequential(
            Linear(self.smallest_dim * 4, self.smallest_dim),
            ReLU(),
            Linear(self.smallest_dim, self.smallest_dim)
        ), aggr='max')

        self.conv4 = EdgeConv(Sequential(
            Linear(self.smallest_dim * 2, self.smallest_dim * 2),
            ReLU(),
            Linear(self.smallest_dim * 2, self.num_features)
        ), aggr='max')

        # Tracking for evaluation
        self.background_test_loss = None
        self.background_train_loss = None
        self.signal_loss = None
        self.train_hist = []
        self.val_hist = []
        self.signal_hist = []
        self.test_data = None
        self.signal_data = None

    def encoder(self, x, edge_index, data, training=True):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

    def decoder(self, x, edge_index, batch, training=True):
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        return x

    def forward(self, data, knn=False, topk=False, training=True):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index, data, training)

        if knn:
            edge_index = knn_graph(x, k=self.num_reduced_edges)

        if topk:
            x, edge_index, _, batch, _, _ = self.topk(x, edge_index.to(torch.int64), batch=data.batch)
        else:
            batch = data.batch

        x = self.decoder(x, edge_index, batch, training)
        return x