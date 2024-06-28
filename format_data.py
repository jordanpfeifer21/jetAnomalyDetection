import pandas as pd 
from data_analysis import make_histogram
import numpy as np 
import torch
from scipy.spatial.distance import cdist
import constants as c
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset


def format_2D(data):
    hist = [make_histogram(data['eta'][i], data['phi'][i],data['pt'][i]) for i in range(data.shape[0])]
    hist = np.reshape(hist, (-1, c.BINS, c.BINS, 1))
    return hist

def load_files(background_file, signal_file):
    background = pd.read_pickle(background_file)
    signal = pd.read_pickle(signal_file) 
    return background, signal 


def make_graph(pt, eta, phi): 
    num_nodes = len(pt) 

    node_pairs = np.column_stack((eta, phi))

    distances = cdist(node_pairs, node_pairs)
    edge1 = [] 
    edge2 = []
    for i in range(num_nodes): 
        closest_indices = np.argsort(distances[i][1:]) # add edges between all nodes 
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

def load_files(background_file, signal_file):
    background = pd.read_pickle(background_file)
    signal = pd.read_pickle(signal_file) 
    return background, signal 

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)

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