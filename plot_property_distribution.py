import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

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

            plt.clf()


    # background_pt = [pt for sublist in background['pt'] for pt in sublist]
    # signal_pt = [pt for sublist in signal['pt'] for pt in sublist]

    # bins = 50  # Number of bins
    

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)
plot_property_distribution(background, signal)
