import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import EdgeConv, TopKPooling, global_max_pool
from torch_cluster import knn_graph


class JetGraphAutoencoderClassification(nn.Module):
    """
    Graph-based Autoencoder for classification tasks on jet data.
    """
    def __init__(self, num_features: int = 3, smallest_dim: int = 8, topk=None, num_reduced_edges: int = 8):
        super(JetGraphAutoencoderClassification, self).__init__()
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
            Linear(self.smallest_dim, self.smallest_dim)
        ), aggr='max')

        # Decoder layers
        self.conv3 = EdgeConv(Sequential(
            Linear(self.smallest_dim * 2, self.smallest_dim),
            ReLU(),
            Linear(self.smallest_dim, self.smallest_dim)
        ), aggr='max')

        self.conv4 = EdgeConv(Sequential(
            Linear(self.smallest_dim * 2, self.smallest_dim * 2),
            ReLU(),
            Linear(self.smallest_dim * 2, self.smallest_dim * 2)
        ), aggr='max')

        # Fully connected layers
        self.fc3 = Linear(self.smallest_dim * 2, 2 * self.num_features)
        self.out = Linear(2 * self.num_features, 1)

    def encoder(self, x, edge_index, data):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

    def decoder(self, x, edge_index, batch):
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = global_max_pool(x, batch)
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

    def forward(self, data, knn=True, topk=True):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index, data)

        if knn:
            edge_index = knn_graph(x, k=self.num_reduced_edges)

        if topk:
            x, edge_index, _, batch, _, _ = self.topk(x, edge_index.to(torch.int64), batch=data.batch)
        else:
            batch = data.batch

        x = self.decoder(x, edge_index, batch)
        return torch.sigmoid(x)
