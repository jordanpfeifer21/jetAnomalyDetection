from format_data_jv import format_2D
from modelsVAE import Autoencoder, train_model, Transformer, SmallAutoencoder, TestAE, TestVAE
from torch.optim import Adam
from torch.nn import MSELoss, CrossEntropyLoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np 
from data_analysis import plot_property_distribution2

from torchinfo import summary
import torch
import constants as c
import pandas as pd 
from model_analysisVAE import plot_loss, plot_anomaly_score, plot_roc
import glob 
import os

background_label = "qcd"
signal_label = "hbb"
props = ['d0Err']
data_dir = "/eos/user/j/jopfeife/2024/processed_data"

batch_size = 25
epochs = 25
initial_lr = 0.001
weight_decay = 1e-3
latent_dim = 12

n_props = len(props)
# n_props = 8
# props = ['-211', '-13', '-11', '11', '13', '22', '130', '211']

prop_string = ''.join(['_' + str(prop) for prop in props])


print("Data Loading ...")

#load background files
pkl_files = glob.glob(os.path.join((data_dir+'/QCD/400to500'), f'*{background_label}*.pkl'))
background = pd.concat([pd.read_pickle(file) for file in pkl_files], ignore_index=True)

print("BACKGROUND LOADED AND SAVED")


#load signal data
pkl_files = glob.glob(os.path.join((data_dir+'/HBB'), f'*{signal_label}*.pkl'))
signal = pd.concat([pd.read_pickle(file) for file in pkl_files], ignore_index=True)

print("FILES LOADED")


background, scalers, background_data = format_2D(background, properties=props, scalers=None)
signal, _, signal_data = format_2D(signal, properties=props, scalers=scalers)

print("Background events: ", len(background))
print("Signal events: ", len(signal))

train_data, test_data = train_test_split(background, test_size = 0.4)

for i, prop in enumerate(props):
    plot_property_distribution2(background_data[i], signal_data[i], prop, background_label, signal_label)




#plot_property_distribution(background, signal, props)

input_shape = train_data.shape
n_props = input_shape[1]
print(input_shape)




X_train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
X_test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
X_valid_dataloader = DataLoader(signal, batch_size=batch_size, shuffle=False)

print("Data Loaded!")

plt.plot

#=============== Define Model =========
print("Input shape: " + str(input_shape))
model = TestVAE(shape = input_shape)
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

print((batch_size, n_props, c.BINS, c.BINS))
summary(model, input_size=(batch_size, n_props, c.BINS, c.BINS))

criterion = MSELoss()
#=============== Run Model =========
print("Training Model... ")
#torch.set_num_threads(3)
train_loss, test_loss, signal_loss = train_model(
    train_dataloader = X_train_dataloader, 
    test_dataloader = X_test_dataloader, 
    signal_dataloader= X_valid_dataloader,
    model = model, 
    loss_fn = MSELoss(), 
    optimizer = optimizer, 
    epochs = epochs, 
    batch_size = batch_size )

#torch.save(model, "model")

#============== Save Analysis========

#plot the latent space if a 2d VAE
def plot_latent_space(model, scale=5.0, n=10, digit_size=32, figsize=15):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            xi = torch.tensor([[[[xi]]]]).to(device)
            yi = torch.tensor([[[[yi]]]]).to(device)
            z_sample = model.reparameterization(xi,yi)

            x_decoded = model.decode(z_sample.float())[0]
            
            
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = 0#digit_size
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.grid(True)
    plt.imshow(figure)
    plt.title('LSP')
    plt.savefig('LSP1')
    plt.clf()



plot_loss(model.train_hist, model.val_hist)
plot_anomaly_score(model.background_anomaly_scores, model.signal_anomaly_scores, background_label, signal_label)
plot_roc(model, signal_label, background_label, examples=True, loss_fn=MSELoss(), properties=props)





# print("TRAIN HIST \n", model.train_hist)
# print("VAL HIST \n", model.val_hist)
print("Done!")
# '''
