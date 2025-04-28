import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd

def plot_loss(train_loss, val_loss, save_path='plots/loss.png'):
    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_anomaly_score(test_scores, anomaly_scores, background_label, signal_label, save_path='plots/anomaly_score.png'):
    plt.figure()
    bins = 100  
    range_ = (min(np.min(anomaly_scores), np.min(test_scores)),
            max(np.max(anomaly_scores), np.max(test_scores)))
    plt.hist(anomaly_scores, bins=bins, range=range_,
            color='red', alpha=0.5, label=f'signal: {signal_label}', density=True)
    plt.hist(test_scores, bins=bins, range=range_,
            color='blue', alpha=0.5, label=f'background: {background_label}', density=True)
    
    plt.xlabel('Loss (MSE)')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(model, signal_label, background_label, savepath, examples, loss_fn, properties=[]):
    
        test_loss = model.background_test_loss
        anomaly_loss = model.signal_loss

        min_val = np.min([np.min(anomaly_loss), np.min(test_loss)])
        max_val = np.max([np.max(anomaly_loss), np.max(test_loss)])

        thresholds = np.linspace(min_val, max_val, num=500)

        e_signal = []
        e_background = []
        for threshold in thresholds:
           
            pred_signal_signal = (test_loss > threshold).flatten()
            pred_signal_background = (anomaly_loss > threshold).flatten()
            true_signal_signal = np.ones_like(pred_signal_signal)
            true_signal_background = np.ones_like(pred_signal_background)

            tp_signal = np.sum(np.logical_and(pred_signal_signal, true_signal_signal))
            tp_background = np.sum(np.logical_and(pred_signal_background, true_signal_background))

            tot_signal = len(true_signal_signal) 
            tot_background = len(true_signal_background)

            e_signal.append(float(tp_signal)/tot_signal)
            e_background.append((float(tp_background)/tot_background))
            
        plt.figure()
        plt.grid()
        plt.tight_layout()
        plt.plot(e_signal, e_background, label='Autoencoder')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel(f'e_signal: {signal_label}')
        plt.ylabel(f'e_background: {background_label}')

        plt.title('Receiver operating characteristic (ROC) curve')
        plt.legend()

        auc_score = auc(e_signal, e_background)
        print('AUC: {:.3f}'.format(auc_score))
        plt.savefig(savepath)
        plt.close()