import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from histogram import make_histogram, plot_histogram

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)

def plot_avg(data, filetype):
    hist_array = []
    for index, row in data.iterrows(): 
        hist_array.append(make_histogram(data['eta'][index], data['phi'][index], data['pT'][index]))
    avg = np.mean(hist_array, axis=0)
    plot_histogram(avg, "plots/avg_"+ filetype + ".png", "average_" + filetype)

plot_avg(background, "background")
plot_avg(signal, "signal")