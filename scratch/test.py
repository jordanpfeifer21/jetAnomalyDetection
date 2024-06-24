import pandas as pd
import numpy as np 
from tqdm import tqdm
import torch 
from scipy.spatial.distance import cdist
import os

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)

# background.insert(len(background.columns) -1, "Dummy Label", np.zeros(background.shape[0]))
# background.to_csv('./dummy_labels.csv')

# data = pd.read_csv('./data/raw/dummy_labels.csv')



from torch_geometric.data import Dataset, Data

class JetDataset(Dataset): 
    def __init__(self, root, transform=None, pretransform=None): 
        self.test = False
        super(JetDataset, self).__init__(root, transform, pretransform)
    
    @property
    def raw_file_names(self): 
        return 'background.pkl'
    
    @property 
    def processed_file_names(self): 
        return 'not_implemented.pt'
    
    def download(self): 
        torch.save(self, f='dataset.pt')
        pass

    def process(self): 
        self.data = pd.read_pickle(self.raw_paths[0])
        for index, jet in tqdm(self.data.iterrows(), total=self.data.shape[0]): 
            node_feats = self.get_node_features(jet)
            # edge_feats = self.get_edge_features(jet)
            edge_index = self.get_adjacency_info(jet)
            # label = self._get_labels(jet['Dummy Label'])
            label = 0

            data = Data(x=node_feats,
                        edge_index=edge_index, 
                        y=label)
            torch.save(data, 
                       os.path.join(self.processed_dir, 
                                    f'data_{index}.pt'))
    
    def get_node_features(self, jet): 
        all_node_feats = [] 
        for i in range(len(jet['pt'])): 
            node_feats = [] 
            # Feature 1: pT
            node_feats.append(list(jet['pt'])[i])
            # Feature 2: XXX 
                # code goes here 
            # Feature 3: YYY 
                # code goes here 

            all_node_feats.append(node_feats)
        all_node_feats = np.asarray(all_node_feats)
        # print(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)
    
    def get_edge_features(self, jet): 
        # all_edge_feats = [] 
        pass # no edge features yet 

    def get_adjacency_info(self, jet):
        eta = jet['eta']
        phi = jet['phi']
        num_nodes = len(eta)
        node_pairs = np.column_stack((eta, phi))

        distances = cdist(node_pairs, node_pairs)
        edge1 = []
        edge2 = []
        for i in range(num_nodes): 
            closest_indices = np.argsort(distances[i][1:11]) # add edges between the 10 closest nodes  
            row = i*np.ones(len(closest_indices))
            edge1.extend(closest_indices)
            edge2.extend(row)
        coo = np.array(list(zip(edge1, edge2)))
        coo = np.reshape(coo, (2, -1)) 
        return torch.tensor(coo, dtype=torch.long)
   
    def len(self): 
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data


dataset = JetDataset(root='data/')

print(dataset[0])


import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(42)
import torch
import torch.nn.functional as F 
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import TransformerConv, GATConv, TopKPooling, BatchNorm
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.conv.x_conv import XConv
torch.manual_seed(42)

class GNN(torch.nn.Module):
    def __init__(self, feature_size):
        super(GNN, self).__init__()
        num_classes = 2
        embedding_size = 1024

        # GNN layers
        self.conv1 = GATConv(feature_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform1 = Linear(embedding_size*3, embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)
        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform2 = Linear(embedding_size*3, embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)
        self.conv3 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.3)
        self.head_transform3 = Linear(embedding_size*3, embedding_size)
        self.pool3 = TopKPooling(embedding_size, ratio=0.2)

        # Linear layers
        self.linear1 = Linear(embedding_size*2, 1024)
        self.linear2 = Linear(1024, num_classes)  

    def forward(self, x, edge_index, batch_index):
        # First block
        x = self.conv1(x, edge_index)
        x = self.head_transform1(x)

        x, edge_index, edge_attr, batch_index, _, _ = self.pool1(x, 
                                                        edge_index, 
                                                        None, 
                                                        batch_index)
        x1 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Second block
        x = self.conv2(x, edge_index)
        x = self.head_transform2(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool2(x, 
                                                        edge_index, 
                                                        None, 
                                                        batch_index)
        x2 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Third block
        x = self.conv3(x, edge_index)
        x = self.head_transform3(x)
        x, edge_index, edge_attr, batch_index, _, _ = self.pool3(x, 
                                                        edge_index, 
                                                        None, 
                                                        batch_index)
        x3 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Concat pooled vectors
        x = x1 + x2 + x3

        # Output block
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)

        return x
    
train_dataset = dataset 
test_dataset = dataset

import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
# from dataset_featurizer import MoleculeDataset
# from model import GNN
import mlflow.pytorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# #%% Loading the dataset
# train_dataset = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")
# test_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv")

#%% Loading the model
model = GNN(feature_size=train_dataset[0].x.shape[1]) 
model = model.to(device)
print(f"Number of parameters: {count_parameters(model)}")
model

#%% Loss and Optimizer
weights = torch.tensor([1, 10], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


#%% Prepare training
NUM_GRAPHS_PER_BATCH = 256
train_loader = DataLoader(train_dataset, 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, 
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

def train(epoch):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)  
        # Reset gradients

        optimizer.zero_grad() 
        # Passing the node features and the connection info
        pred = model(batch.x.float(), 
                                batch.edge_index, 
                                batch.batch) 
        # Calculating the loss and gradients
        loss = torch.sqrt(loss_fn(pred, batch.y)) 
        loss.backward()  
        # Update using the gradients
        optimizer.step()  

        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return loss

def test(epoch):
    all_preds = []
    all_labels = []
    for batch in test_loader:
        batch.to(device)  
        pred = model(batch.x.float(), 
                        batch.edge_index, 
                        batch.batch) 
        loss = torch.sqrt(loss_fn(pred, batch.y))    
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return loss


def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_pred, y_true)}")
    print(f"Accuracy: {accuracy_score(y_pred, y_true)}")
    print(f"Precision: {precision_score(y_pred, y_true)}")
    print(f"Recall: {recall_score(y_pred, y_true)}")
    try:
        roc = roc_auc_score(y_pred, y_true)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")

# %% Run the training
with mlflow.start_run() as run:
    for epoch in range(3):
        # Training
        model.train()
        loss = train(epoch=epoch)
        loss = loss.detach().cpu().numpy()
        print(f"Epoch {epoch} | Train Loss {loss}")
        mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

        # Testing
        model.eval()
        if epoch % 5 == 0:
            loss = test(epoch=epoch)
            loss = loss.detach().cpu().numpy()
            print(f"Epoch {epoch} | Test Loss {loss}")
            mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)

        scheduler.step()
    print("Done.")


# %% Save the model 
mlflow.pytorch.log_model(model, "model")