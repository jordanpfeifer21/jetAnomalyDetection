from fast_histogram import histogram2d
import numpy as np 
import matplotlib.pyplot as plt 

def make_histogram(eta, phi, cc):
    eta_min, eta_max = -0.8, 0.8
    phi_min, phi_max = -0.8, 0.8
    incr = 0.05

    hist_range = [[eta_min, eta_max], [phi_min, phi_max]]
    eta_bins = np.arange(eta_min, eta_max, incr)
    phi_bins = np.arange(phi_min, phi_max, incr)
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
      

