import torch
import torch.cuda
from torch import cuda
from torchvision import datasets, transforms
from torch.nn import Sequential, Conv2d, Linear, AvgPool2d, Flatten, ReLU, Upsample, Unflatten, ConvTranspose2d, MSELoss
from torch import Tensor
from torch.utils.data import DataLoader
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from torch import reshape
import torch.nn as nn 
from tqdm import tqdm
import constants as c

#regular autoencoder
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

#experimenting with smaller model Autoencoders
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

#The tets AE is changed and benchmarked against the above AE
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
        #print("the channel size: ", str(adj_var))
        self.encoder = nn.Sequential(
            # Layer 1: input (adj_var, 32, 32), output (adj_var*10, 16, 16)
            nn.Conv2d(in_channels=adj_var, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            
            # Layer 2: input (adj_var*10, 16, 16), output (adj_var*10, 8, 8)
            nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True), 

            
            # Layer 5: input (adj_var*10, 8, 8), output (adj_var, 6, 6)
            nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var, kernel_size=3, stride=1, padding=0),
           
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: input (adj_var, 6, 6), output (adj_var*10, 8, 8)
            nn.ConvTranspose2d(in_channels=adj_var, out_channels=adj_var*10, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),
            
    
            # Layer 3: input (adj_var*10, 8, 8), output (adj_var*10, 16, 16)
            nn.ConvTranspose2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # Layer 4: input (adj_var*10, 16, 16), output (adj_var, 32, 32)
            nn.ConvTranspose2d(in_channels=adj_var*10, out_channels=adj_var, kernel_size=3, stride=2, padding=1, output_padding=1),
          
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#the test VAE is modified and benchmarked against the AE
class TestVAE(nn.Module):
    def __init__(self, shape):
        super(TestVAE, self).__init__()
        self.shape = shape
        self.shape0 = shape[2]
        alpha_init = np.random.randn()
        self.background_test_loss = None 
        self.background_train_loss = None
        self.signal_loss = None 
        self.background_anomaly_scores = None
        self.signal_anomaly_scores = None
        self.train_hist = [] 
        self.val_hist = []
        self.test_data = None 
        self.signal_data = None
        alpha_init = np.random.randn()

        adj_var = shape[1]
        self.in_channels = adj_var
        
        self.encoder = nn.Sequential(
            # Layer 1: input (adj_var, 32, 32), output (adj_var*10, 16, 16)
            nn.Conv2d(in_channels=adj_var, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            
            # Layer 2: input (adj_var*10, 16, 16), output (adj_var*10, 8, 8)
            nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var, kernel_size=3, stride=2, padding=1),
            #nn.ReLU(True), 

	    #Layer 3: input (adj_var*10, 8,8), output (adj_var, 6,6)
            nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var, kernel_size=3, stride=2, padding=1),
            
            
        )
        
         # Latent space
        self.mean_layer = nn.Linear(8,8)#nn.Conv2d(in_channels=adj_var*1, out_channels=adj_var, kernel_size=3, stride=1, padding=0)#nn.Linear(6*6,  6*6)
        self.logvar_layer = nn.Linear(8,8)#nn.Conv2d(in_channels=adj_var*1, out_channels=adj_var, kernel_size=3, stride=1, padding=0)#nn.Linear(6*6 ,  6*6)
        
        
        # Decoder
        self.decoder = nn.Sequential(
       	    
	    #Layer 1: input (adj_var, 6,6 ), output (adj_var*10, 8, 8) 
            nn.ConvTranspose2d(in_channels=adj_var, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),    
            #Layer 2: input (adj_var*10, 8, 8), output (adj_var*10, 16, 16)
            nn.ConvTranspose2d(in_channels=adj_var, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            
            # Layer 3: input (adj_var*10, 16, 16), output (adj_var, 32, 32)
            nn.ConvTranspose2d(in_channels=adj_var*10, out_channels=adj_var, kernel_size=3, stride=2, padding=1, output_padding=1),
          
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)    
        z = mean + var*epsilon
        return epsilon* torch.exp(var*0.5) + mean

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = x.float()
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    

#A 2d VAE for visualizing the latent space
class TestVAE2d(nn.Module):
    def __init__(self, shape):
        super(TestVAE2d, self).__init__()
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
        self.in_channels = adj_var
        
        self.encoder = nn.Sequential(
            # Layer 1: input (adj_var, 32, 32), output (adj_var*10, 16, 16)
            nn.Conv2d(in_channels=adj_var, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            
            # Layer 2: input (adj_var*10, 16, 16), output (adj_var*10, 8, 8)
            nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True), 

            
            # Layer 3: input (adj_var*10, 8, 8), output (adj_var*10, 6, 6)
            nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=1, padding=0),
           
            # Layer 4: input (adj_var*10, 6, 6), output (adj_var*10, 4, 4)
            nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=1, padding=0),
            
	    #Layer 5: input (adj_var*10, 4,4), output (adj_var*10, 2,2)
	    nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=1, padding=0),
            
            #Layer 5: input (adj_var*10, 2, 2), output (adj_var*2, 1, 1)
            nn.Conv2d(in_channels=adj_var*10, out_channels=adj_var*2, kernel_size=2, stride=1, padding=0)
        )
        
        # Latent space
        self.mean_layer = nn.Linear(1,1)#nn.Conv2d(in_channels=adj_var*1, out_channels=adj_var, kernel_size=3, stride=1, padding=0)#nn.Linear(6*6,  6*6)
        self.logvar_layer = nn.Linear(1,1)#nn.Conv2d(in_channels=adj_var*1, out_channels=adj_var, kernel_size=3, stride=1, padding=0)#nn.Linear(6*6 ,  6*6)
        
        self.decoder = nn.Sequential(
            
            # Layer 1: input (adj_var*2, 1, 1), output (adj_var*10, 2, 2)
            nn.ConvTranspose2d(in_channels=adj_var*2, out_channels=adj_var*10, kernel_size=2, stride=1, padding=0),
            nn.ReLU(True),
            
	    #Layer 2: input (adj_var*10, 2,2), output (adj_var*10, 4,4 )
            nn.ConvTranspose2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),

	    #Layer 3: input (adj_var*10, 4, 4), output (adj_var*10, 6, 6)
            nn.ConvTranspose2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),

	    #Layer4: input (adj_var*10, 6,6) output (adj_var, 8, 8)
            nn.ConvTranspose2d(in_channels=adj_var*10, out_channels=adj_var*10, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),

            # Layer 5: input (adj_var*10, 8, 8), output (adj_var*10, 16, 16)
            nn.ConvTranspose2d(in_channels=adj_var*10, out_channels=adj_var, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.ReLU(True),
            
            # Layer 6: input (adj_var*10, 16, 16), output (adj_var, 32, 32)
            nn.ConvTranspose2d(in_channels=adj_var*10, out_channels=adj_var, kernel_size=3, stride=2, padding=1, output_padding=1),
          
        )
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)    
        z = mean + var*epsilon
        return epsilon* torch.exp(var*0.5) + mean

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = x.float()
        mean, logvar = self.encode(x)
        
        z = self.reparameterization(mean, logvar)
        
        x_hat = self.decode(z)
        
        return x_hat, mean, logvar
    


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
        #print(x_flat)
        encoded = self.encoder_linear(x_flat)  # (batch_size, latent_dim)
        encoded = encoded.unsqueeze(1)  # Add sequence length dimension (batch_size, seq_len=1, latent_dim)  
        # Transformer Encoder
        encoded = self.transformer_encoder(encoded)  # (batch_size, seq_len=1, latent_dim)
        # Decode
        decoded = self.decoder_linear(encoded.squeeze(1))  # (batch_size, prod(input_size)) 
        decoded = decoded.view(x.shape)
        #print("The mean of decoded: " +  str(torch.mean(decoded[0][0][0])))
        return decoded


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = []
    max_mean = None
    min_mean = None

    max_var = None
    min_var = None
    for batch, X in enumerate(dataloader):
        # Compute prediction and loss
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = X.to(device)


        beta = 0
        pred, mean, logvar = model(X.float())

        kl_loss_og = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)

        kl_loss = kl_loss_og.cpu()
        
        kl_loss = torch.sum(kl_loss) 
        recon_loss = loss_fn(pred, X).cpu()
        
        
                
        loss = (recon_loss*(1-beta))+(kl_loss*beta)
        
        

        total_loss.append(float(loss))
        
        
        optimizer.zero_grad()  # Clear the gradients
        loss.backward()     # Compute the gradients
        optimizer.step()  
def eval_loop(dataloader, model, loss_fn, test=False, signal=False):
    model.eval()
    loss = []
    data = []
    anomaly_scores = []
    if (test == True) or (signal == True):
        latent_x = []
        latent_y = []
    with torch.no_grad():
        for X in dataloader:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            X = X.to(device)
            beta = 0
            pred, mean, logvar = model(X.float())
            mean = torch.mean(mean)
            
            pred = pred.cpu()
            if (test == True) or (signal == True):
                latent_x = pred[0]
                latent_y = pred[1]
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            
            kl_loss = torch.sum(kl_loss) 
            
            X = X.cpu()
            recon_loss = loss_fn(pred, X)
            loss_app = (recon_loss*(1-beta))+(kl_loss*beta)
            loss.append(float(loss_app))
            data.append(X)
            kl_loss = kl_loss.cpu()
            anomaly_scores.append(kl_loss)
    if test: 
        model.test_data = data 
    elif signal: 
        model.signal_data = data
    
    
    return loss, anomaly_scores

def train_model(train_dataloader, test_dataloader, signal_dataloader, model, loss_fn, optimizer, epochs, batch_size):
    for epoch in tqdm(range(epochs)):
        train_loss = []
        val_loss = []
        signal_loss = []
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss, test_as =(eval_loop(test_dataloader, model, loss_fn, test=True, signal=False)) 
        signal_loss, signal_as = (eval_loop(signal_dataloader, model, loss_fn, test=False, signal=True))
        val_loss.extend(test_loss)

        train_loss.extend((eval_loop(train_dataloader, model, loss_fn, test=False, signal=False))[0])
        signal_loss.extend(signal_loss)

        model.train_hist.append(np.mean(train_loss))
        model.val_hist.append(np.mean(val_loss))

        #for plotting latent space with 2d VAE
	'''
        if (epoch == 0) or (epoch%5 == 0):
            plt.scatter(test_lx, test_ly, label="Test")
            plt.scatter(signal_lx, signal_ly, label="Signal")
            #plt.plot(val_loss, label = "Validation Loss")
            plt.legend()
            #plt.xlabel('')
            #plt.ylabel('Loss')
            plt.title('Latent Space at Epoch: ' + str(epoch))
            plt.savefig('LSV' +str(epoch+1))
            plt.clf() 
        '''


    model.background_test_loss = val_loss
    model.background_train_loss = train_loss
    model.signal_loss = signal_loss
    model.background_anomaly_scores = test_as
    model.signal_anomaly_scores = signal_as
    return train_loss, val_loss, signal_loss
