from dgl.data import DGLDataset
import dgl.nn.pytorch as dglnn
import torch.nn as nn 
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
import numpy as np 
import pandas as pd 
import dgl
from scipy.spatial.distance import cdist
import torch 
from dgl.nn import GraphConv
from dgl.dataloading import GraphDataLoader 
from format_data import MyDataset

dataset = MyDataset()
dataloader = GraphDataLoader(dataset, batch_size = 1, shuffle=True)
print("data loaded!")

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

model = GCN(dataset.dim_nfeats, 3, dataset.gclasses)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(2):
    # print('epoch', epoch)
    for batched_graph, labels in dataloader:
        labels = labels.long()
        pred = model(batched_graph, batched_graph.ndata['feat'].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct = 0
num_tests = 0
for batched_graph, labels in dataloader:
    pred = model(batched_graph, batched_graph.ndata['feat'].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print('Test accuracy:', num_correct / num_tests)