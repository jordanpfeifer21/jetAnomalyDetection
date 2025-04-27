import torch
import numpy as np
from torch_geometric.data import Data
from torch_cluster import knn_graph
import pandas as pd
from typing import List

def make_graph(data: dict,
               data_label: int,
               node_feature_names: List[str] = ['pt', 'eta', 'phi', 'd0/d0Err', 'dz/dzErr'],
               nearest_neighbors: int = 16,
               device: str = 'cuda') -> Data:
    """
    Build a graph from an event's particle information.
    """
    x = torch.tensor(np.column_stack([data[name] for name in node_feature_names]), dtype=torch.float).to(device)
    nn_features = torch.tensor(np.column_stack([data['eta'], data['phi']]), dtype=torch.float).to(device)
    edge_index = knn_graph(nn_features, k=nearest_neighbors, loop=False).to(device)
    y = torch.tensor([data_label], dtype=torch.long).to(device)

    return Data(x=x, edge_index=edge_index, y=y)

def graph_data_loader(df: pd.DataFrame,
                      data_label: int,
                      nearest_neighbors: int = 16,
                      device: str = 'cuda') -> List[Data]:
    """
    Convert a dataframe of events into a list of graphs.
    """
    graphs = []
    for i in range(len(df['eta'].tolist())):
        graphs.append(make_graph(df.iloc[i], data_label=data_label, nearest_neighbors=nearest_neighbors, device=device))
    return graphs