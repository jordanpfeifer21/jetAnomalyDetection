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
import torch.nn as nn 
from tqdm import tqdm
import constants as c

class Autoencoder(torch.nn.Module):
    def __init__(self, shape, latent_dim = 12):
        super(Autoencoder, self).__init__()

        self.shape = shape
        self.shape0 = shape[2]
        alpha_init = np.random.randn()
        self.background_test_loss = None 
        self.background_train_loss = None
        self.signal_loss = None 
        self.train_hist = [] 
        self.val_hist = []
        self.test_data = None 
        self.signal_data = None

        self.encoder = Sequential(
            Conv2d(self.shape[1], 10, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Conv2d(10, 5, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1),
            Conv2d(5, 5, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Conv2d(5, 5, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Flatten(),
            Linear(5*17*self.shape[3], latent_dim),
            ReLU(alpha_init)
        )

        self.decoder = Sequential(
            Linear(latent_dim, 24),
            ReLU(alpha_init),
            Linear(24, c.BINS * 2 * self.shape[3]),
            ReLU(alpha_init),
            Unflatten(1, (8, 8, self.shape[3])),  
            Conv2d(8, 2*c.BINS, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Upsample(scale_factor=(4,1)),
            Conv2d(2*c.BINS, c.BINS, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Conv2d(c.BINS, c.BINS, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class SmallAutoencoder(torch.nn.Module):
    def __init__(self, shape, latent_dim = 12):
        super(SmallAutoencoder, self).__init__()

        self.shape = shape
        print(shape)
        self.shape0 = shape[2]
        alpha_init = np.random.randn()
        self.background_test_loss = None 
        self.background_train_loss = None
        self.signal_loss = None 
        self.train_hist = [] 
        self.val_hist = []
        self.test_data = None 
        self.signal_data = None

        self.encoder = Sequential(
            Conv2d(self.shape[1], 5, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Conv2d(5, 2, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=1),
            Flatten(),
            Linear(17*2*self.shape[3], latent_dim),
            #Linear(68, latent_dim),

            ReLU(alpha_init)
        )

        self.decoder = Sequential(
            Linear(latent_dim, c.BINS * 2 * self.shape[3]),
            ReLU(alpha_init),

            Unflatten(1, (8, 8, self.shape[3])),  
            Conv2d(8, c.BINS, kernel_size=(4,4), padding='same'),
            ReLU(alpha_init),
            Upsample(scale_factor=(4,1)),

        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
class TestAE(nn.Module):
    
    def __init__(self, shape):
        super(TestAE, self).__init__()

        self.shape = shape
        self.shape0 = shape[2]
        alpha_init = np.random.randn()
        self.background_test_loss = None 
        self.background_train_loss = None
        self.signal_loss = None 
        self.train_hist = [] 
        self.val_hist = []
        self.test_data = None 
        self.signal_data = None
        alpha_init = np.random.randn()

        adj_var = shape[1]
        self.encoder = nn.Sequential(
            # Layer 1: input (32, 32, adj_var), output (16, 16, adj_var*10)
            nn.Conv2d(in_channels=adj_var, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            
            # Layer 2: input (16, 16, adj_var*10), output (8, 8, adj_var*40)
            nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            

            
            # Layer 5: input (8, 8, adj_var*10), output (6, 6, adj_var)
            nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var, kernel_size=3, stride=1, padding=0),
           
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: input (6, 6, adj_var), output (8, 8, adj_var*10)
            nn.ConvTranspose2d(in_channels=adj_var, out_channels=adj_var*10, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),
            
            
            
            # Layer 3: input (8, 8, adj_var*20), output (16, 16, adj_var*10)
            nn.ConvTranspose2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # Layer 4: input (16, 16, adj_var*10), output (32, 32, adj_var)
            nn.ConvTranspose2d(in_channels=adj_var*10, out_channels=adj_var, kernel_size=3, stride=2, padding=1, output_padding=1),
          
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class TestVAE(nn.Module):
    
    def __init__(self, shape):
        super(TestVAE, self).__init__()

        self.shape = shape
        self.shape0 = shape[2]
        alpha_init = np.random.randn()
        self.background_test_loss = None 
        self.background_train_loss = None
        self.signal_loss = None 
        self.train_hist = [] 
        self.val_hist = []
        self.test_data = None 
        self.signal_data = None
        alpha_init = np.random.randn()

        in_channels = shape[1]
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 5, kernel_size=3, stride=2, padding=1), # (32, 32, in_channels) -> (16, 16, in_channels)
            nn.ReLU(alpha_init),
            nn.Conv2d(5, 5, kernel_size=3, stride=2, padding=1), # (16, 16, in_channels) -> (8, 8, in_channels)
            nn.ReLU(alpha_init),
            nn.Conv2d(5, 1, kernel_size=3, stride=2, padding=1), # (8, 8, in_channels) -> (4, 4, in_channels)
            
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 5, kernel_size=3, stride=2, padding=1, output_padding=1), # (4, 4, in_channels) -> (8, 8, in_channels)
            nn.ReLU(alpha_init),
            nn.ConvTranspose2d(5, 5, kernel_size=3, stride=2, padding=1, output_padding=1), # (8, 8, in_channels) -> (16, 16, in_channels)
            nn.ReLU(alpha_init),
            nn.ConvTranspose2d(5, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # (16, 16, in_channels) -> (32, 32, in_channels)
          
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
class Transformer(nn.Module):
    # Autoencoder with attention.
    def __init__(self, input_size, latent_dim=12, num_heads=2, num_layers=2):
        super().__init__()
        
        self.shape = input_size
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.background_test_loss = None 
        self.background_train_loss = None
        self.signal_loss = None 
        self.train_hist = [] 
        self.val_hist = []
        self.test_data = None 
        self.signal_data = None
        data_size = np.prod(input_size[1:])

        # Encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dropout=0.2, activation='relu', dim_feedforward=36)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        # Decoder layers
        decoder_layers = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=num_heads, dropout=0.2, activation='relu', dim_feedforward=36)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        # Linear layers for encoding and decoding
        self.encoder_linear = nn.Linear(data_size, latent_dim)
        self.decoder_linear = nn.Linear(latent_dim, data_size)

    def forward(self, x):
        # Flatten input
        x = x.float()
        x_flat = x.view(x.size(0), -1)  # Reshape to (batch_size, input_size)     
        # x_flat = [i.float() for i in x_flat]  
        # Encode
        # print(x_flat.shape)
        encoded = self.encoder_linear(x_flat)  # (batch_size, latent_dim)
        encoded = encoded.unsqueeze(1)  # Add sequence length dimension (batch_size, seq_len=1, latent_dim)  
        # Transformer Encoder
        encoded = self.transformer_encoder(encoded)  # (batch_size, seq_len=1, latent_dim)
        # Decode
        decoded = self.decoder_linear(encoded.squeeze(1))  # (batch_size, prod(input_size)) 
        decoded = decoded.view(x.shape)
        return decoded


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = []
    for batch, X in enumerate(dataloader):
        # Compute prediction and loss
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = X.to(device)
        pred = model(X)
        loss = loss_fn(pred, X)
        total_loss.append(float(loss))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    


def eval_loop(dataloader, model, loss_fn, test=False, signal=False):
    model.eval()
    loss = []
    data = []

    with torch.no_grad():
        for X in dataloader:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            X = X.to(device)
            pred = model(X)
            #print(pred.shape)
            loss.append(float(loss_fn(pred, X)))
            #print(X.shape)
            data.append(X)
    if test: 
        model.test_data = data 
        
    elif signal: 
        model.signal_data = data

    return loss

def train_model(train_dataloader, test_dataloader, signal_dataloader, model, loss_fn, optimizer, epochs, batch_size):
    for epoch in tqdm(range(epochs)):
        # print(f"Epoch [{epoch+1}/{epochs}]")
        train_loss = []
        val_loss = []
        signal_loss = []
        train_loop(train_dataloader, model, loss_fn, optimizer)
        val_loss.extend((eval_loop(test_dataloader, model, loss_fn, test=True, signal=False)))
        train_loss.extend((eval_loop(train_dataloader, model, loss_fn, test=False, signal=False)))
        signal_loss.extend((eval_loop(signal_dataloader, model, loss_fn, test=False, signal=True)))

        model.train_hist.append(np.mean(train_loss))
        model.val_hist.append(np.mean(val_loss))

    model.background_test_loss = val_loss
    model.background_train_loss = train_loss
    model.signal_loss = signal_loss
    return train_loss, val_loss, signal_loss
