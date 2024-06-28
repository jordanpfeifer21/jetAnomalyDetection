
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd 
import numpy as np 
from scipy.spatial.distance import cdist
import torch 

def load_files(background_file, signal_file): 
    return pd.read_pickle(background_file), pd.read_pickle(signal_file)

def make_graph(data, signal=False): 
    eta = data['eta']
    phi = data['phi']

    node_feature_names = ['pt', 'd0', 'dz', 'mass', 'charge', 'pdgId']
    node_features = data[node_feature_names]
    node_features = np.column_stack(node_features)

    num_nodes = len(eta)
    node_pairs = np.column_stack((eta, phi))
    distances = cdist(node_pairs, node_pairs)
    edges1 = [] 
    edges2 =[]
    edge_attributes = []
    for i in range(num_nodes): 
        closest_indices = np.argsort(distances[i][1:])
        closest_indices = [j for j in closest_indices if j >= i]
        edges1.extend(closest_indices)
        edges2.extend(i * np.ones(len(closest_indices)))
        edge_attributes.extend([distances[i][j] for j in closest_indices])

    edge_index = torch.tensor([edges1, edges2], dtype=torch.int)
    edge_attributes = torch.tensor(np.reshape(np.array(edge_attributes), (-1, 1)), dtype=torch.float64)
    # print((edge_attributes).shape)
    node_data = torch.tensor(np.reshape(node_features,(-1,len(node_feature_names))), dtype=torch.float64)
    if signal: 
        label = 1
    else: 
        label = 0

    g = Data(x=node_data, edge_index=edge_index, edge_attr=edge_attributes, y=label)
    return g  

def graph_data_loader(data): 
    graphs = [] 
    for d in data: 
        for i in range(len(d['eta'].tolist())): 
            graphs.append(make_graph(d.iloc[i]))
    return graphs 

class JetDataset(InMemoryDataset): 
    def __init__(self, data_list, root=None, transform=None): 
        self.data_list = data_list 
        super().__init__(root, transform)
        self.load(self.processed_paths[0])
    
    @property 
    def processed_file_names(self): 
        return 'data.pt'
    
    def process(self): 
        self.save(self.data_list, self.processed_paths[0])


