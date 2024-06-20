import pandas as pd 
from data_analysis import make_histogram
import numpy as np 
import dgl #deep graph network that will make integrating graphs easier
import torch
from scipy.spatial.distance import cdist
import constants as c
import dgl.nn.pytorch as dglnn
import torch.nn as nn 
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset


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

    g = dgl.DGLGraph()

    node_pairs = np.column_stack((eta, phi))

    distances = cdist(node_pairs, node_pairs)
    for i in range(num_nodes): 
        closest_indices = np.argsort(distances[i][1:2]) # add edges between the 10 closest nodes 
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



class MyDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir="",
                 force_reload=False,
                 verbose=False):
        super(MyDataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
        self.dim_nfeats = 2
        self.gclasses = 2

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        # process raw data to graphs, labels, splitting masks
        self.graphs = get_graphs([background, signal])
        self.labels = np.hstack([np.zeros(100), np.ones(20)])

    def __getitem__(self, idx):
        # get one example by index
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        # number of data examples
        return len(self.graphs)

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass