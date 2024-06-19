import pandas as pd 
from histogram import make_histogram
import numpy as np 
import dgl #deep graph network that will make integrating graphs easier
import torch
from scipy.spatial.distance import cdist

def format_2D(data):
    hist = [make_histogram(data['eta'][i], data['phi'][i],data['pT'][i]) for i in range(data.shape[0])]
    hist = np.reshape(hist, (-1, 32, 32, 1))
    return hist

def format_graph(data): 
    pt = data['pT'].tolist()
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

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background, signal = load_files(background_file, signal_file)

two_dimensional_background, two_dimensional_signal = format_2D(background), format_2D(signal)
graph_background, graph_signal = format_graph(background), format_graph(signal)


def make_graph(data): 

    # src_ids = torch.tensor([2, 3, 4]) # source nodes for edges
    # dst_ids = torch.tensor([1, 2, 3]) # destination nodes 
    # # edges are now (2, 1), (3, 2), (4, 3)
    # g = dgl.graph((src_ids, dst_ids), num_nodes=100)
    # print(g)
    # return g 
    pt = data['pT'].tolist()[0]
    eta = data['eta'].tolist()[0]
    phi = data['phi'].tolist()[0]
    num_nodes = len(pt)


    g = dgl.DGLGraph()

    node_pairs = np.column_stack((eta, phi))
    # node_pairs = np.reshape((node_pairs), (len(pt), 2))
    print(node_pairs.shape)

    distances = cdist(node_pairs, node_pairs)
    for i in range(num_nodes): 
        closest_indices = np.argsort(distances[i][1:10]) # add edges between the 10 closest nodes 
        g.add_edges(i, closest_indices)
        g.add_edges(closest_indices, i)
    print(len(pt))
    pt = np.reshape(pt, (-1, 1))
    print(pt.shape)
    feat = torch.tensor(node_pairs)
    g.ndata['feat'] = feat 
    g.ndata['pt'] = torch.tensor(pt)


    print(g)
    return g 

make_graph(background)
