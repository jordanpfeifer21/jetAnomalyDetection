import pandas as pd
import numpy as np 
from fast_histogram import histogram2d
import matplotlib.pyplot as plt
from histogram import plot

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background = pd.read_pickle(background_file)
signal = pd.read_pickle(signal_file)

hist = plot(background['eta'][1], background['phi'][1], background['pT'][1])
plt.imshow(hist)
plt.savefig("plots/ex_hist.png")