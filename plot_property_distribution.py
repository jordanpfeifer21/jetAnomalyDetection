import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

def plot_property_distribution(background, signal):
    fig, ax = plt.subplots()
    ax.set_title("Distribution of pT")
    ax.set_xlabel("pT")

    background_pt = [pt for sublist in background['pT'] for pt in sublist]
    signal_pt = [pt for sublist in signal['pT'] for pt in sublist]

    bins = 50  # Number of bins
    bin_range = (np.min([np.min(background_pt), np.min(signal_pt)]),
                    np.max([np.max(background_pt), np.max(signal_pt)]))
    
    ax.hist(background_pt, density=True, label='Background', alpha=0.5, bins=bins, range = bin_range)
    ax.hist(signal_pt, density=True, label='Signal', alpha=0.5, bins=bins, range = bin_range)
    plt.legend()
    plt.savefig("plots/pt_distribution.png")

    plt.clf()

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)
plot_property_distribution(background, signal)
