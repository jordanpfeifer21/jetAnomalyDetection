import matplotlib.pyplot as plt
import numpy as npfrom sklearn.metrics import auc
from sklearn.svm import SVC




def graph_loss(model):
    plt.plot(model.train_loss, label="Training Loss")
    plt.plot(model.val_loss, label "Validation Loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.savefig('Training and Validation Losses')
    plt.clf()

def plot_anomaly_score(model):
    plt.figure()
    bins = 50  
    range_ = (min(np.min(model.anomaly_scores), np.min(model.test_scores)),
            max(np.max(model.anomaly_scores), np.max(model.test_scores)))

    weights_anomaly = np.ones_like(model.anomaly_scores.flatten()) / len(model.anomaly_scores.flatten())
    weights_test = np.ones_like(model.test_scores.flatten()) / len(model.test_scores.flatten())

    plt.hist(model.anomaly_scores.flatten(), bins=bins, range=range_, weights=weights_anomaly,
            color='red', alpha=0.5, label='anomalous', density=False)
    plt.hist(model.test_scores.flatten(), bins=bins, range=range_, weights=weights_test,
            color='blue', alpha=0.5, label='not anomalous', density=False)

    plt.xlabel('Anomaly Score')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.savefig('Anomaly Score Distribution')
    plt.clf()