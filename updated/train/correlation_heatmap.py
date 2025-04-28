import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

def plot_feature_correlation(df, anomaly_scores, feature_names):
    """
    Plots a correlation heatmap between features and the anomaly score.
    """
    # Flatten feature arrays into a matrix
    feature_matrix = np.stack([np.concatenate(df[name]) for name in feature_names], axis=1)

    # Repeat anomaly scores so it matches feature matrix length
    anomaly_scores_expanded = np.repeat(anomaly_scores, feature_matrix.shape[0] // len(anomaly_scores))

    df_corr = pd.DataFrame(feature_matrix, columns=feature_names)
    df_corr['anomaly_score'] = anomaly_scores_expanded

    corr = df_corr.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation with Anomaly Score")
    plt.tight_layout()
    plt.show()
    plt.clf()
