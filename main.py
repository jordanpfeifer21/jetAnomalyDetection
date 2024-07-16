from format_data import format_2D
from models import Autoencoder, train_model, Transformer, SmallAutoencoder, TestAE
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np 
from data_analysis import plot_property_distribution
# from torchsummary import summary
from torchinfo import summary
import torch
import constants as c
import pandas as pd 
from model_analysis import plot_loss, plot_anomaly_score, plot_roc
import glob 
import os

background_label = "qcd"
signal_label = "wjet"
props = ['pdgId']
data_dir = "/eos/user/j/jopfeife/2024/processed_data"

batch_size = 15
epochs = 25
initial_lr = 0.001
weight_decay = 1e-3
latent_dim = 12

n_props = len(props)
# n_props = 8
# props = ['-211', '-13', '-11', '11', '13', '22', '130', '211']

prop_string = ''.join(['_' + str(prop) for prop in props])

'''
background_file = data_dir + "/background.pkl"
signal_file = data_dir + "/signal.pkl"

background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)
# plot_property_distribution(background, signal, props)

background = format_2D(background, properties=props)
np.save(data_dir + "/background" + prop_string + ".npy", background)
print("BACKGROUND LOADED AND SAVED")

signal = format_2D(signal, properties=props)
np.save(data_dir + "/signal" + prop_string + ".npy", signal)
print("SIGNAL LOADED AND SAVED")

'''
print("Data Loading ...")
'''
background = np.load(data_dir + "/" + prop_string + ".npy", allow_pickle=True).reshape(-1, 32, 32, n_props)
signal = np.load(data_dir + "/signal" + prop_string + ".npy", allow_pickle=True).reshape(-1, 32, 32, n_props)
'''
#print("Background path: ", (data_dir+'/QCD/400to500'), f'*{background_label}*.pkl')
pkl_files = glob.glob(os.path.join((data_dir+'/QCD/400to500'), f'*{background_label}*.pkl'))
#print("Background files: ", pkl_files)
background = pd.concat([pd.read_pickle(file) for file in pkl_files], ignore_index=True)
background = format_2D(background, props)

pkl_files = glob.glob(os.path.join((data_dir+'/WJET/400to500'), f'*{signal_label}*.pkl'))
signal = pd.concat([pd.read_pickle(file) for file in pkl_files], ignore_index=True)
signal = format_2D(signal, props)
print("FILES LOADED")

print("Background events: ", len(background))
print("Signal events: ", len(signal))
train_data, test_data = train_test_split(background, test_size = 0.4)

input_shape = train_data.shape
n_props = input_shape[1]
print(input_shape)

X_train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
X_test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
X_valid_dataloader = DataLoader(signal, batch_size=batch_size, shuffle=False)

print("Data Loaded!")

plt.plot

#=============== Define Model =========
# model = Autoencoder(input_shape)
# model =SmallAutoencoder(input_shape, latent_dim=latent_dim)
# optimizer = Adam(model.parameters(), lr = initial_lr)

print("Input shape: " + str(input_shape))
model = TestAE(shape = input_shape)
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

print((batch_size, n_props, c.BINS, c.BINS))
summary(model, input_size=(batch_size, n_props, c.BINS, c.BINS))
#print("Input shape:" + str(input_shape))
#summary(model, input_size = input_shape)
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
print("Analyzing Results... ")
plot_loss(model.train_hist, model.val_hist)
plot_anomaly_score(model.background_test_loss, model.signal_loss, background_label, signal_label)
plot_roc(model, signal_label, background_label, examples=True, loss_fn=MSELoss(), properties=props)

# print("TRAIN HIST \n", model.train_hist)
# print("VAL HIST \n", model.val_hist)
print("Done!")
# '''
