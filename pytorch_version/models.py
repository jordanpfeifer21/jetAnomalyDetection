import torch
from torchvision import datasets, transforms
from torch.nn import Sequential, Conv2d, Linear, AvgPool2d, Flatten, ReLU, Upsample
from torch.utils.data import DataLoader
import numpy as np



class Autoencoder(torch.nn.Module):
    def __init__(self, shape, latent_dim = 12):
        super().__init__()
        self.shape = shape, 
        self.n_properties - shape[2]
        #self.latent_dim = latent_dim

        self.encoder = Sequential(
            Conv2d(out_channels = 10, kernel_size = (4,4), padding = 'same'),
            ReLU(),
            Conv2d(out_channels = 5, kernel_size = (4,4), padding = 'same'),
            ReLU(),
            AvgPool2d(pool_size = (2,2), strides = (2,2), padding='same'),
            Conv2d(out_channels = 5, kernel_size = (4,4), padding = 'same'),
            ReLU(),
            Conv2d(out_channels = 5, kernel_size = (4,4), padding = 'same'),
            ReLU(),
            Conv2d(out_channels = 5, kernel_size = 8, padding = 'same'),
            ReLU(),
            Flatten(),
            Linear(out_features = latent_dim),
            ReLU(),
        )

        self.decoder = Sequential(
            Linear(latent_dim,out_features = 100),
            ReLU(),
            Linear(out_features = 64),
            ReLU(),
            torch.reshape((8,8,1)),
            Conv2D(out_channels = 5, kernel_size = (4,4), padding = 'same'),
            ReLU(),
            Upsample(size = (2,2)),
            ConvTranspose2d(out_channels = self.n_properties, kernel_size = (4,4), padding = 'same')

        )
    
    def forward(self, x):
        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded

def train_loop(dataloader, model, loss_func, optimizer, batch_size):

    device = torch.device("cuda" if torch.cuda.is_avaliabe() else "cpu")
    model.to(device)
    model.train()
    train_loss = 0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    for batch, X in enumerate(dataloader):
        X = X.to(device)
        pred = model(X)
        loss = loss_func(pred, X)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    return train_loss

def test_loop(dataloader, model, loss_function):

    device = torch.device("cuda" if torch.cuda.is_avaliabe() else "cpu")
    model.to(device)
    model.eval()
    test_loss = 0

    with torch.no_grad(): # No gradient calculated.
        for X in dataloader: # for (X,y) in ... for labelled data.
            X = X.to(device)
            pred = model(X)
            test_loss += loss_func(pred, X).item() # X -> y
    test+loss /= num_batches

    return(test_loss)

def train_model(dataloader_train, dataLoader_test, model, loss_func, optimizer, epochs, scheduler = None, batch_size, graph_path=None):
    train_loss = np.zeros(epochs)
    test_losst = np.zeros(epochs)

    for t in np.arrange(0, epochs, step =1):
        train_loss[t] = train_loop(dataloader_train, model, loss_func, optimizer, batch_size=batch_size)
        test_loss[t] = test_loop(dataLoader_test, model, loss_func)

        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            print(f"Current learning rate: {current_lr[0]}")
    
    return train_loss, test_loss


