from format_data import format_2D
from models import Autoencoder, train_model, Transformer, SmallAutoencoder
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np 
from data_analysis import plot_property_distribution, plot_property_distribution2
# from torchsummary import summary
from torchinfo import summary
import torch
import constants as c
import pandas as pd 
from model_analysis import plot_loss, plot_anomaly_score, plot_roc
import glob 
import os

background_label = "qcd400to500"
signal_label = "wjet400to500"
props = ['pt']
data_dir = "/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data"

batch_size = 20
epochs = 8
initial_lr = 0.001
weight_decay = 1e-3
latent_dim = 12

n_props = len(props)
prop_string = ''.join(['_' + str(prop) for prop in props])

# n_props = 8
# props = ['-211', '-13', '-11', '11', '13', '22', '130', '211']

print("BINS", c.BINS)

'''

pkl_files = glob.glob(os.path.join(data_dir, f'*{background_label}*.pkl'))
background = pd.concat([pd.read_pickle(file) for file in set(pkl_files)], ignore_index=True)
print(len(background))

pkl_files = glob.glob(os.path.join(data_dir, f'*{signal_label}*.pkl'))
signal = pd.concat([pd.read_pickle(file) for file in set(pkl_files)], ignore_index=True)
print(len(signal))
print("FILES LOADED")

background, scalers, background_data, fractions_background = format_2D(background, properties=props, scalers=None)
np.save(data_dir + "/" + background_label + prop_string + ".npy", background)
print("BACKGROUND LOADED AND SAVED")

signal, _, signal_data, fractions_signal = format_2D(signal, properties=props, scalers=scalers)
np.save(data_dir + "/" + signal_label + prop_string + ".npy", signal)
print("SIGNAL LOADED AND SAVED")

# props = ["-211", "-13", "-11", "11", "13", "22", "130", "211"]
# print(fractions_background, fractions_signal)
# plot_property_distribution2(fractions_background, fractions_signal, 'fraction of PFCands kept with new binning strategy', background_label, signal_label)

for i, prop in enumerate(props): 
    plot_property_distribution2(background_data[i], signal_data[i], prop, background_label, signal_label)


'''
print("Data Loading ...")
background = np.load(data_dir + "/" + background_label + prop_string + ".npy", allow_pickle=True).reshape(-1, c.BINS, c.BINS, n_props)
signal = np.load(data_dir + "/" + signal_label + prop_string + ".npy", allow_pickle=True).reshape(-1, c.BINS, c.BINS, n_props)

print("Background events: ", len(background))
print("Signal events: ", len(signal))
train_data, test_data = train_test_split(background, test_size = 0.2)
input_shape = train_data.shape
n_props = input_shape[-1]
print(input_shape)

X_train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
X_test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
X_valid_dataloader = DataLoader(signal, batch_size=batch_size, shuffle=False)

print("Data Loaded!")


#=============== Define Model =========
# model = Autoencoder(input_shape)
# model =SmallAutoencoder(input_shape, latent_dim=latent_dim)
# optimizer = Adam(model.parameters(), lr = initial_lr)

model = Transformer(input_size = input_shape, latent_dim=latent_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

summary(model, input_size=(batch_size, c.BINS, c.BINS, n_props))

criterion = MSELoss(reduction='mean')

#=============== Run Model =========
print("Training Model... ")
torch.set_num_threads(3)
train_loss, test_loss, signal_loss = train_model(
    train_dataloader = X_train_dataloader, 
    test_dataloader = X_test_dataloader, 
    signal_dataloader= X_valid_dataloader,
    model = model, 
    loss_fn = MSELoss(reduction='mean'), 
    optimizer = optimizer, 
    epochs = epochs, 
    batch_size = batch_size )

torch.save(model, "model")

#============== Save Analysis========
print("Analyzing Results... ")
plot_loss(model.train_hist, model.val_hist)
plot_anomaly_score(model.background_test_loss, model.signal_loss, background_label, signal_label)
plot_roc(model, signal_label, background_label, examples=True, loss_fn=MSELoss(reduction='mean'), properties=props)
print("Done!")
# '''