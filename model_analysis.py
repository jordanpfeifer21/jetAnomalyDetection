import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.svm import SVC
import constants as c
from data_analysis import plot_histogram
import torch 
from models import eval_loop
from torch.nn import MSELoss
from matplotlib.colors import TwoSlopeNorm


def process_data(data_list, index, loss, loss_type, loss_fn, model, is_max, properties):
    num_m = len(data_list[index])
    figs = {}
    axes = {}
    for prop in properties:
        figs[prop], axes[prop] = plt.subplots(2, num_m, figsize=(num_m * 5, 10))

    # loss_func = np.argmax if is_max else np.argmin
    loss_value = np.max(loss) if is_max else np.min(loss)
    
    for i, m in enumerate(data_list[index]):
        m = m.reshape(-1, model.shape[1], c.BINS, c.BINS)
        reco = model(m)[0]
        reco = reco.cpu()

        for j, prop in enumerate(properties):
            m_prop = m[0, j, :, :] # Select channel for the current property
            m_prop = m_prop.cpu() 
            reco_prop = reco[j, :, :]  # Corresponding reconstructed property
            if torch.min(m_prop) < 0.0: 
                norm = TwoSlopeNorm(vcenter=0.0)
            else: 
                norm = TwoSlopeNorm(vcenter=(np.mean([torch.max(m_prop),torch.min(m_prop)])))
            # Original
            axes[prop][0, i].imshow(m_prop, norm=norm)
            axes[prop][0, i].axis('off')
            

            # Reconstructed
            axes[prop][1, i].imshow(reco_prop, norm=norm)
            axes[prop][0, i].set_title(f"loss: {loss_fn(m, model(m))}")
            axes[prop][1, i].axis('off')
    
    for prop in properties:
        plt.figure(figs[prop].number)  # Activate the figure by its number
        # plt.title(f"{loss_type} {prop}: {loss_value}")
        plt.suptitle(f"{loss_type} {prop}: {loss_value}", fontsize=num_m)
        # plt.tight_layout()
        figs[prop].savefig(f"plots/example_loss/{loss_type}_{prop}.png")

    plt.show()
    plt.clf()

def plot_examples(model, loss_fn, signal_label, background_label, properties=[]): 
        loss_fn = MSELoss()
        test_loss = model.background_test_loss 
        anomaly_loss = model.signal_loss
        test_data_list = model.test_data
        anomaly_data_list = model.signal_data

        # Convert losses to numpy arrays for easier manipulation
        test_loss = np.array(test_loss)
        anomaly_loss = np.array(anomaly_loss)

        # Identify max and min loss indices
        test_max_index = np.argmax(test_loss)
        anomaly_max_index = np.argmax(anomaly_loss)
        test_min_index = np.argmin(test_loss)
        anomaly_min_index = np.argmin(anomaly_loss)

        # Retrieve corresponding data
        with torch.no_grad():                          
                process_data(test_data_list, test_max_index, test_loss, f"maximum test ({background_label}) loss", loss_fn, model, is_max=True, properties=properties)
                process_data(test_data_list, test_min_index, test_loss, f"minimum test ({background_label}) loss", loss_fn, model, is_max=False, properties=properties)
                process_data(anomaly_data_list, anomaly_max_index, anomaly_loss, f"maximum anomaly ({signal_label}) loss", loss_fn, model, is_max=True, properties=properties)
                process_data(anomaly_data_list, anomaly_min_index, anomaly_loss, f"minimum anomaly ({signal_label}) loss", loss_fn, model, is_max=False, properties=properties)

def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label = "Validation Loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.savefig('plots/Training and Validation Losses')
    plt.clf()

def plot_anomaly_score(test_scores, anomaly_scores, background_label, signal_label):    
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
    plt.savefig('plots/Anomaly Score Distribution')
    plt.show()
    plt.clf()




def plot_roc(model, signal_label, background_label, examples=False, loss_fn=MSELoss(), properties=[]):
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
        plt.plot(e_signal, e_background, label='Autoencoder')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel(f'e_signal: {signal_label}')
        plt.ylabel(f'e_background: {background_label}')

        plt.title('Receiver operating characteristic (ROC) curve')
        plt.legend()
        plt.savefig("plots/ROC.png")
        plt.clf()

        auc_score = auc(e_signal, e_background)
        print('AUC: {:.3f}'.format(auc_score))

        if examples: 
               plot_examples(model, loss_fn, signal_label, background_label, properties=properties)
