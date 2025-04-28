from scipy.stats import norm
import numpy as np 
import matplotlib.pyplot as plt 

def calculate_bump_pvalues(hist_counts, bin_edges):
    background_estimate = np.median(hist_counts)  # crude estimate
    residuals = hist_counts - background_estimate
    z_scores = residuals / np.sqrt(background_estimate)
    p_values = norm.sf(z_scores)  # one-sided
    return p_values


def plot_bump_pvalues(background_scores, signal_scores, bins=100):
    """
    Plots p-values from a simple bump hunt between background and signal.
    """
    combined_scores = np.concatenate([background_scores, signal_scores])

    # Histogram anomaly scores
    hist_bg, bin_edges = np.histogram(background_scores, bins=bins)
    hist_sig, _ = np.histogram(signal_scores, bins=bin_edges)

    # Assume background expectation is median
    background_expectation = np.median(hist_bg)
    residuals = hist_sig - background_expectation

    # Avoid divide-by-zero
    errors = np.sqrt(np.maximum(background_expectation, 1.0))
    z_scores = residuals / errors
    p_values = norm.sf(z_scores)  # one-sided p-value

    # Plot
    plt.figure(figsize=(10, 6))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(centers, p_values, marker='o', linestyle='-')
    plt.yscale('log')
    plt.xlabel('Anomaly Score')
    plt.ylabel('p-value')
    plt.title('Bump Hunt p-values (Signal Excess)')
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.clf()