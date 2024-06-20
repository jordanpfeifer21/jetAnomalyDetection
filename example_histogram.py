import pandas as pd
import numpy as np 
from histogram import make_histogram, plot_histogram

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
                


