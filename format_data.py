import pandas as pd 
from histogram import make_histogram
import numpy as np 
import dgl #deep graph network that will make integrating graphs easier
import torch
from scipy.spatial.distance import cdist
import constants as c
import dgl.nn.pytorch as dglnn
import torch.nn as nn 
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader


def format_2D(data):
    hist = [make_histogram(data['eta'][i], data['phi'][i],data['pt'][i]) for i in range(data.shape[0])]
    hist = np.reshape(hist, (-1, c.BINS, c.BINS, 1))
    return hist

def format_graph(data): 
    pt = data['pt'].tolist()
    eta = data['eta'].tolist()
    phi = data['phi'].tolist()
    
    graph = []
    for i in range(len(pt)): 
        nodes = []
        for j in range(len(pt[i])):
            nodes.append([pt[i][j], eta[i][j], phi[i][j]])
        nodes = np.reshape(nodes, (-1, 3))
        graph.append(nodes)

    return graph 

def load_files(background_file, signal_file):
    background = pd.read_pickle(background_file)
    signal = pd.read_pickle(signal_file) 
    return background, signal 

# background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
# signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
# background, signal = load_files(background_file, signal_file)

# two_dimensional_background, two_dimensional_signal = format_2D(background), format_2D(signal)
# graph_background, graph_signal = format_graph(background), format_graph(signal)


def make_graph(pt, eta, phi): 
    num_nodes = len(pt)

    g = dgl.DGLGraph()

    node_pairs = np.column_stack((eta, phi))

    distances = cdist(node_pairs, node_pairs)
    for i in range(num_nodes): 
        closest_indices = np.argsort(distances[i][1:c.CLOSEST_NEIGHBORS]) # add edges between the 10 closest nodes 
        g.add_edges(i, closest_indices)
        g.add_edges(closest_indices, i)
    pt = np.reshape(pt, (-1, 1))
    feat = torch.tensor(node_pairs)
    g.ndata['feat'] = feat # is this better than individual eta and phi? what exactly is feat? 
    g.ndata['pt'] = torch.tensor(pt)
    # print(g)
    return g 

def get_graphs(data):
    graphs = []
    for i in range(len(data['eta'].tolist())): 
        pt = data['pt'][i]
        eta = data['eta'][i]
        phi = data['phi'][i]
        graphs.append(make_graph(pt, eta, phi))
    return graphs 

# graphs = get_graphs(background) # list of graphs 
# bg = dgl.batch(graphs) # combines graphs together 


class Classifier(nn.Module): 
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)

# dataset = bg
# # labels = np.zeroS
# data = dgl.load_graphs("./data.bin", idx_list=None)
# print(data)
    
# model = Classifier(7, 20, 5)
# opt = torch.optim.Adam(model.parameters())
# for epoch in range(20):
#     for batched_graph, labels in dataloader:
#         feats = batched_graph.ndata['attr']
#         logits = model(batched_graph, feats)
#         loss = F.cross_entropy(logits, labels)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()

