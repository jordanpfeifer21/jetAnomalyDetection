import torch
import torch.cuda
from torch import cuda
from torchvision import datasets, transforms
from torch.nn import Sequential, Conv2d, Linear, AvgPool2d, Flatten, ReLU, Upsample, Unflatten, ConvTranspose2d, MSELoss
from torch import Tensor
from torch.utils.data import DataLoader
import torch.utils.data
import numpy as np

from torchsummary import summary
from torch import reshape




class Autoencoder(torch.nn.Module):
    def __init__(self, shape, latent_dim = 12):
        super().__init__()
        
        self.anomaly_scores = None
        self.test_scores = None

        self.shape = shape

        self.n_properties = shape[2]
        alpha_init = np.random.randn()
        self.encoder = Sequential(
            Conv2d(self.n_properties, 10, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Conv2d(10, 5, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1),
            Conv2d(5, 5, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Conv2d(5, 5, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Flatten(),
            Linear(5*17, latent_dim),
            ReLU(alpha_init)
        )
        self.decoder = Sequential(
            Linear(latent_dim, 24),
            ReLU(alpha_init),
            Linear(24, 64),
            ReLU(alpha_init),
            Unflatten(1, (8, 8, 1)),  
            Conv2d(8, 64, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Upsample(scale_factor=(4,1)),
            Conv2d(64, 32, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Conv2d(32, 32, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_loop(dataloader, model, loss_func, optimizer, batch_size):

    device = torch.device("cuda" if cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    train_loss = 0
    size = dataloader.size
    num_batches = len(dataloader)

    X = torch.from_numpy(dataloader)
    X = X.to(device)
   
    pred = model(X)
    print("Shapes")
    print(X.shape)
    print(pred.shape)
    loss = loss_func(pred, X)
    


    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    train_loss = loss.item()/num_batches

    return train_loss

def test_loop(dataloader, model, loss_function):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #summary(model, (32,32,1))
    model.eval()
    test_loss = 0

    #with torch.no_grad(): # No gradient calculated.
        #for X in dataloader: # for (X,y) in ... for labelled data.
    X = torch.from_numpy(dataloader)
    X = X.to(device)
    pred = model(X)

    test_loss += loss_function(pred, X).item() # X -> y
    test_loss /= len(dataloader)

    return(test_loss)

def train_model(dataloader_train, dataLoader_test, model, loss_func, optimizer, epochs, batch_size, scheduler = None,  graph_path=None):
    
 
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    anomaly_scores = np.zeros(epochs)
    

    for t in np.arange(0, epochs, step =1):
        
        train_loss[t] = train_loop(dataloader_train, model, loss_func, optimizer, batch_size=batch_size)
        test_loss[t] = test_loop(dataLoader_test, model, loss_func)
        #anomaly_scores[t] = 
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            print(f"Current learning rate: {current_lr[0]}")
    
    return train_loss, test_loss


