from format_data_jv import format_2D
from model5 import Autoencoder, train_model, Transformer, SmallAutoencoder, TestAE, TestVAE2d, TestVAE
from torch.optim import Adam
from torch.nn import MSELoss, CrossEntropyLoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np 
from data_analysis import plot_property_distribution2
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
props = ['d0Err']
data_dir = "/eos/user/j/jopfeife/2024/processed_data"

batch_size = 25
epochs = 5
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

#np.save(data_dir + "/" + background_label + prop_string + ".npy", background)
print("BACKGROUND LOADED AND SAVED")

#background = format_2D(background, props)

pkl_files = glob.glob(os.path.join((data_dir+'/WJET/400to500'), f'*{signal_label}*.pkl'))
signal = pd.concat([pd.read_pickle(file) for file in pkl_files], ignore_index=True)

#np.save(data_dir + "/" + signal_label + prop_string + ".npy", signal)
#signal = format_2D(signal, props)
print("FILES LOADED")

#background, scalers, background_data_flat = format_2D(background, properties=props, scalers=None)
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
# model = Autoencoder(input_shape)
# model =SmallAutoencoder(input_shape, latent_dim=latent_dim)
# optimizer = Adam(model.parameters(), lr = initial_lr)

print("Input shape: " + str(input_shape))
model = TestVAE(shape = input_shape)
optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

print((batch_size, n_props, c.BINS, c.BINS))
summary(model, input_size=(batch_size, n_props, c.BINS, c.BINS))
#print("Input shape:" + str(input_shape))
#summary(model, input_size = input_shape)
criterion = CrossEntropyLoss()
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

def plot_latent_space(model, scale=5.0, n=10, digit_size=32, figsize=15):
    # display a n*n 2D manifold of digits
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
            
            #torch.tensor([[[xi, yi]]], dtype=torch.float).to(device)
            #print("Z-sample: " + str(z_sample))
            #print(z_sample.size)
            x_decoded = model.decode(z_sample.float())[0]
            #print("x mean: " + str(torch.mean(x_decoded)))
            #print(x_decoded.shape)
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


'''
def plot_latent_space(model, scale=10.0, n=6, digit_size=6, figsize=15):
    # display a n*n 2D manifold of digits
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid 
    grid_x = np.linspace(-scale, 20, n)
    grid_y = np.linspace(-20, 20, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            mean = torch.tensor([[[xi, xi, xi, xi, xi, xi], [xi, xi, xi, xi, xi, xi],[xi, xi, xi, xi, xi, xi],[xi, xi, xi, xi, xi, xi],[xi, xi, xi, xi, xi, xi],[xi, xi, xi, xi, xi, xi]]], dtype=torch.float).to(device)
            logvar = torch.tensor([[[yi, yi, yi, yi, yi, yi],[yi, yi, yi, yi, yi, yi],[yi, yi, yi, yi, yi, yi],[yi, yi, yi, yi, yi, yi],[yi, yi, yi, yi, yi, yi],[yi, yi, yi, yi, yi, yi]]], dtype=torch.float).to(device)
            z_sample = model.reparameterization(mean, logvar)
            #print(z_sample)
            #print(z_sample.shape)
            #z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            #print(x_decoded)
            #digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            digit = z_sample[0].detach().cpu().reshape(digit_size, digit_size)
            #print(figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size])
      
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size] = digit
    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = 0
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.grid(True)
    print(figure.shape)
    plt.imshow(figure)
    #plt.figure(figsize=(figsize, figsize))
    plt.title('LSP')
    plt.savefig('LSP')
    plt.show()
    plt.clf()

def plot_latent_space(model, scale=10.0, n=25, digit_size=28, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid 
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size,] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
def plot_latent_space(vae, n=6, figsize=15):
    # Create a grid of latent vectors
    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]

    # Initialize a figure
    figure = np.zeros((32 * n, 32 * n))

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float32)
            with torch.no_grad():
                x_decoded = vae.decode(z_sample).numpy()
            digit = x_decoded[0, 0]
            figure[i * 32: (i + 1) * 32,
                   j * 32: (j + 1) * 32] = digit

    plt.figure(figsize=(figsize, figsize))
    
    plt.title('LSP')
    plt.savefig('LSP')
    plt.show()
    plt.clf()
'''
print("Analyzing Results... ")

#plot_latent_space(model)

plot_loss(model.train_hist, model.val_hist)
plot_anomaly_score(model.background_test_anomaly_scores, model.signal_anomaly_scores, background_label, signal_label)
plot_roc(model, signal_label, background_label, examples=True, loss_fn=MSELoss(), properties=props)





# print("TRAIN HIST \n", model.train_hist)
# print("VAL HIST \n", model.val_hist)
print("Done!")
# '''
