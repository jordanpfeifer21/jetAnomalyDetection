import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from fast_histogram import histogram2d
import constants as c 

def make_histogram(eta, phi, cc):
    hist_range = [[c.ETA_MIN, c.ETA_MAX], [c.PHI_MIN, c.PHI_MAX]]
    eta_bins = np.arange(c.ETA_MIN, c.ETA_MAX, c.INCR)
    phi_bins = np.arange(c.PHI_MIN, c.PHI_MAX, c.INCR)
    image_shape = (eta_bins.shape[0], phi_bins.shape[0])

    return histogram2d(phi, eta, range=hist_range, bins=image_shape, weights=cc)

def plot_histogram(hist, save_file_name, title): 
    fig, ax = plt.subplots() 
    ax.imshow(hist)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])  
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$\eta$')
    ax.set_title(title)  
    plt.setp(ax.spines.values(), alpha = 0)
    plt.savefig(save_file_name)
    plt.close(fig)

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)

idx = 1
eta = background['eta'][idx]
phi = background['phi'][idx]

for prop_name in background.columns: 
        if prop_name != "eta" and prop_name != "phi":
            background_prop = background[prop_name][idx]
            if len(background_prop) == len(eta): 
                hist = make_histogram(eta, phi, background_prop)
                plot_histogram(hist, "plots/histograms/ex_hist_" + prop_name + ".png", "example background " + prop_name)
                
def plot_avg(data, filetype):
    eta = data['eta']
    phi = data['phi']
    
    for prop_name in data.columns: 
        if prop_name != "eta" and prop_name != "phi":
            if len(data[prop_name][0]) == len(eta[0]):
                hist_array = []
                for index, row in data.iterrows(): 
                    hist_array.append(make_histogram(eta[index], phi[index], data[prop_name][index]))
                avg = np.mean(hist_array, axis=0)
                plot_histogram(avg, "plots/histograms/avg_"+ filetype + "_" + prop_name + ".png", "average_" + filetype + "_" + prop_name)
            
plot_avg(background, "background")
plot_avg(signal, "signal")

def plot_property_distribution(background, signal):
    for prop_name in background.columns: 
        if prop_name != "eta" and prop_name != "phi":
            background_prop = [p for sublist in background[prop_name] for p in sublist]
            signal_prop = [p for sublist in signal[prop_name] for p in sublist]

            fig, ax = plt.subplots()
            ax.set_title("Distribution of " + prop_name)
            ax.set_xlabel(prop_name)

            num_unique = len(np.unique(np.concatenate([background_prop, signal_prop])))
            if num_unique < 50: 
                bins = num_unique * 5
            else: 
                bins = 50
        
            bin_range = (np.min([np.min(background_prop), np.min(signal_prop)]),
                    np.max([np.max(background_prop), np.max(signal_prop)]))
            
            ax.hist(background_prop, density=True, label='Background', alpha=0.5, bins=bins, range = bin_range)
            ax.hist(signal_prop, density=True, label='Signal', alpha=0.5, bins=bins, range = bin_range)
            plt.legend()
            plt.savefig("plots/property_distributions/" + prop_name + "_distribution.png")

            plt.close(fig)

plot_property_distribution(background, signal)

