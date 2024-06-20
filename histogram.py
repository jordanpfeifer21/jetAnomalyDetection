from fast_histogram import histogram2d
import numpy as np 
import matplotlib.pyplot as plt 
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
      

