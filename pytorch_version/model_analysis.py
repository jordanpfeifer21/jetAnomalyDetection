import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.svm import SVC
import torch

#find the mse    
def mse(model, test_data, anomaly_data): 
    test_data =  torch.from_numpy(test_data)
    anomaly_data = torch.from_numpy(anomaly_data)

    reconstructed_anomaly = model(anomaly_data) #pass known anomaly data into model
    anomaly_data = anomaly_data.detach().numpy()
    reconstructed_anomaly = reconstructed_anomaly.detach().numpy()
    anomaly_loss = np.mean(np.square(anomaly_data - reconstructed_anomaly), axis=(1,2,3)) #compute the anomaly scores

    #pass known background data into the model
    reconstructed_test = model(test_data)
    test_data = test_data.detach().numpy()
    reconstructed_test = reconstructed_test.detach().numpy()
    test_loss = np.mean(np.square(test_data - reconstructed_test), axis=(1,2,3))

    model.anomaly_scores = anomaly_loss
    model.test_scores = test_loss
    #print mse over anomolous and non anomolus data
    print('anomaly MSE (loss) over all anomalous inputs: ', np.mean(model.anomaly_scores)) 
 
    print('not anomaly MSE (loss) over all non-anomalous inputs: ', np.mean( model.test_scores))

def plot_anomaly_score_distribution(model):
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
        plt.savefig('/isilon/export/home/rpankaj/jetAD/jetAnomalyDetection/Anomaly Score Distribution')
        plt.clf()

def plot_roc(model, test_data, anomaly_data):
        test_data =  torch.from_numpy(test_data)
        anomaly_data = torch.from_numpy(anomaly_data)

        all_data = np.concatenate((test_data, anomaly_data), axis=0)

        all_data = torch.from_numpy(all_data)
        all_data = all_data.detach().numpy()

        data_pred = model(all_data)
        data_pred = data_pred.detach().numpy()

        data_loss = np.mean(np.square(all_data - data_pred), axis=(1,2,3))
        thresholds = np.linspace(np.min(data_loss[len(test_data):]), np.max(data_loss[:len(test_data)]), num=500)
        tprs = []
        fprs = []
        for threshold in thresholds:
            pred_signal = (data_loss > threshold)
            true_signal = np.ones_like(pred_signal)
            true_signal[:len(test_data)] = 0  # The first len(light_data) events are background, the rest are signal
            tp = np.sum(np.logical_and(pred_signal, true_signal))
            fp = np.sum(np.logical_and(pred_signal, 1 - true_signal))
            tn = np.sum(np.logical_and(1 - pred_signal, 1 - true_signal))
            fn = np.sum(np.logical_and(1 - pred_signal, true_signal))
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            tprs.append(tpr)
            fprs.append(fpr)

        plt.figure()
        plt.plot(tprs, fprs, label='Autoencoder')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver operating characteristic (ROC) curve')
        plt.legend()
        plt.savefig("/isilon/export/home/rpankaj/jetAD/jetAnomalyDetection/pytorch_version/ROC")
        plt.clf()

        auc_score = auc(fprs, tprs)
        print('AUC: {:.3f}'.format(auc_score))
def graph_loss(model):
    plt.plot(model.train_loss, label="Training Loss")
    plt.plot(model.val_loss, label ="Validation Loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.savefig('Training and Validation Losses')
    plt.clf()

