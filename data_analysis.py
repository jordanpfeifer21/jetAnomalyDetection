import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from fast_histogram import histogram2d
import constants as c 
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
from matplotlib.colors import TwoSlopeNorm


def make_histogram(eta, phi, cc):
    hist_range = [[c.ETA_MIN, c.ETA_MAX], [c.PHI_MIN, c.PHI_MAX]]
    eta_bins = np.arange(c.ETA_MIN, c.ETA_MAX, c.INCR)
    phi_bins = np.arange(c.PHI_MIN, c.PHI_MAX, c.INCR)
    image_shape = (eta_bins.shape[0], phi_bins.shape[0])

    return histogram2d(phi, eta, range=hist_range, bins=image_shape, weights=cc)

def make_histogram_highest_pt_only(eta, phi, prop, vals_to_keep=None): 
    phi_bins = np.arange(c.PHI_MIN, c.PHI_MAX, c.INCR) # x axis
    eta_bins = np.arange(c.ETA_MIN, c.ETA_MAX, c.INCR) # y axis
    count = 0 
    if vals_to_keep is None: 
        vals_to_keep = []
        hist = np.full((len(phi_bins), len(eta_bins)), 0.0) # start with a histogram of zeros
        hist_indices = np.full((len(phi_bins), len(eta_bins)), np.inf) 
        for index, (phi_val, eta_val, prop_val) in enumerate(zip(phi, eta, prop)): 
            phi_bin = np.digitize(phi_val, phi_bins) - 1
            eta_bin = np.digitize(eta_val, eta_bins) - 1
            if np.abs(hist[phi_bin, eta_bin]) < np.abs(prop_val): 
                count += 1 if hist[phi_bin, eta_bin] == 0.0 else 0
                hist[phi_bin, eta_bin] = prop_val 
                hist_indices[phi_bin, eta_bin] = index
                
        vals_to_keep.extend([item for sublist in hist_indices for item in sublist if item != np.inf])

    else: 
        hist = histogram2d(phi[vals_to_keep], eta[vals_to_keep], bins=(eta_bins.shape[0], phi_bins.shape[0]), weights=prop[vals_to_keep])

    return hist, vals_to_keep
        
def plot_histogram(hist, save_file_name, title, scale = False, scale_label=""): 
    if np.min(hist) < 0.0: 
        norm = TwoSlopeNorm(vcenter=0.0)
    else: 
        norm = None

    fig, ax = plt.subplots() 
    im = ax.imshow(hist, norm=norm)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])  
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$\eta$')
    ax.set_title(title)  
    plt.setp(ax.spines.values(), alpha = 0)

    if scale: 
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(scale_label)

    plt.savefig(save_file_name)
    plt.close(fig)

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
                plot_histogram(avg, "avg_"+ filetype + "_" + prop_name + ".png", "average_" + filetype + "_" + prop_name)

def plot_property_distribution(background, signal, props):
    for prop_name in props: 
        background_prop = [p for sublist in background[prop_name] for p in sublist]
        signal_prop = [p for sublist in signal[prop_name] for p in sublist]
        if prop_name == 'dzErr': 
            background_prop = [p / z for sublist1, sublist2 in zip(background['dz'], background['dzErr']) for p, z in zip(sublist1, sublist2)]
            signal_prop = [p / z for sublist1, sublist2 in zip(signal['dz'], signal['dzErr']) for p, z in zip(sublist1, sublist2)]

        fig, ax = plt.subplots()
        ax.set_title("Distribution of " + prop_name )
        ax.set_xlabel(prop_name)

        num_unique = len(np.unique(np.concatenate([background_prop, signal_prop])))
        if num_unique < 50: 
            bins = num_unique * 5
        else: 
            bins = 150
    
        bin_range = (np.min([np.min(background_prop), np.min(signal_prop)]),
                np.max([np.max(background_prop), np.max(signal_prop)]))

        if not np.isinf(np.min([np.min(background_prop), np.min(signal_prop)])): 
            if not np.isinf(np.max([np.max(background_prop), np.max(signal_prop)])):
                ax.hist(background_prop, density=True, label='Background', alpha=0.5, bins=bins, range = bin_range)
                ax.hist(signal_prop, density=True, label='Signal', alpha=0.5, bins=bins, range = bin_range)
                plt.legend()
                plt.savefig("plots/" + prop_name + "_distribution.png")

                plt.close(fig)

def plot_property_distribution2(background_data, signal_data, prop_name, background_label, signal_label):
    fig, ax = plt.subplots()
    ax.set_title("Distribution of " + prop_name )
    ax.set_xlabel(prop_name)

    bins = 150

    bin_range = (np.min([np.min(background_data), np.min(signal_data)]),
            np.max([np.max(background_data), np.max(signal_data)]))
    

    ax.hist(background_data, density=True, label=background_label, alpha=0.5, bins=bins, range=bin_range)
    ax.hist(signal_data, density=True, label=signal_label, alpha=0.5, bins=bins, range=bin_range)
    plt.legend()
    # plt.yscale('log')
    plt.savefig("plots/" + prop_name + "_distribution.png")
    plt.show()
    plt.close(fig)