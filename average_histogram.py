import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from histogram import make_histogram, plot_histogram

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)

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