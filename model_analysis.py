import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.metrics import auc


def loss(model):
    plt.plot(model.train_loss, label='Training Loss')
    plt.plot(model.val_loss, label='Validation Loss')
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


def roc(model, test_data, anomaly_data):
        qcd_data = test_data
        Hbb_data = anomaly_data

        data_pred_qcd = self.model.predict(qcd_data)
        data_pred_Hbb = self.model.predict(Hbb_data)

        print("77777777777777777777777777777")
        print(type(test_data))
        print(type(anomaly_data))
        print(type(data_pred_qcd))
        print(type(data_pred_Hbb))
        print("77777777777777777777777777777")
        
        data_loss_qcd = np.mean(np.square(qcd_data - data_pred_qcd), axis=(1,2,3))
        data_loss_Hbb = np.mean(np.square(Hbb_data - data_pred_Hbb), axis=(1,2,3))

        min_val = np.min([np.min(data_loss_Hbb), np.min(data_loss_qcd)])
        max_val = np.max([np.max(data_loss_Hbb), np.max(data_loss_qcd)])

        thresholds = np.linspace(0.0000, 0.015, num=500)
        tprs = []
        fprs = []
        e_qcd = []
        e_Hbb = []
        for threshold in thresholds:
           
            pred_signal_qcd = (data_loss_qcd > threshold)
            pred_signal_Hbb = (data_loss_Hbb > threshold)


            true_signal_qcd = np.ones_like(pred_signal_qcd)
            true_signal_Hbb = np.ones_like(pred_signal_Hbb)

            #true_signal_qcd[:len(test_data)] = 0  # The first len(light_data) events are background, the rest are signal
            #true_signal_Hbb[:len(anomaly_data)] = 0  # The first len(light_data) events are background, the rest are signal
           
            tp_qcd = np.sum(np.logical_and(pred_signal_qcd, true_signal_qcd))
            tp_Hbb = np.sum(np.logical_and(pred_signal_Hbb, true_signal_Hbb))

            tot_qcd = len(true_signal_qcd) 
            tot_Hbb = len(true_signal_Hbb)

            e_qcd.append(float(tp_qcd)/tot_qcd)
            e_Hbb.append(1-(float(tp_Hbb)/tot_Hbb))


            '''
            fp = np.sum(np.logical_and(pred_signal, 1 - true_signal))
            tn = np.sum(np.logical_and(1 - pred_signal, 1 - true_signal))
            fn = np.sum(np.logical_and(1 - pred_signal, true_signal))
            tpr = tp / (tp + fn)
            fpr = (fp / (fp + tn))
            tprs.append(tpr)
            fprs.append(fpr)
            '''
        print(e_Hbb)
        print(e_qcd)
        plt.figure()
        plt.plot(e_qcd, e_Hbb, label='Autoencoder')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('e_qcd')
        plt.ylabel('e_Hbb')
        plt.title('Receiver operating characteristic (ROC) curve')
        plt.legend()
        plt.savefig("/isilon/export/home/rpankaj/jetAD/jetAnomalyDetection/Ekin/Figures/ROC")
        plt.clf()

        auc_score = auc(e_Hbb, e_qcd)
        print('AUC: {:.3f}'.format(auc_score))