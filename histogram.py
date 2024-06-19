from fast_histogram import histogram2d
import numpy as np 

def plot(x, y, cc):
    eta_min, eta_max = -0.8, 0.8
    phi_min, phi_max = -0.8, 0.8
    incr = 0.05

    hist_range = [[eta_min, eta_max], [phi_min, phi_max]]
    eta_bins = np.arange(eta_min, eta_max, incr)
    phi_bins = np.arange(phi_min, phi_max, incr)
    image_shape = (eta_bins.shape[0], phi_bins.shape[0])

    return histogram2d(y, x, range=hist_range, bins=image_shape, weights=cc)