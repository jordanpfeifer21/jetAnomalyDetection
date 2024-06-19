import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, UpSampling2D, AveragePooling2D, Conv2DTranspose
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import auc
from tensorflow.keras.activations import relu

def reshape_data(file_name, n_properties): 
    final_data = []
    raw_data = np.load(file_name, allow_pickle=True)
    raw_data = np.reshape(raw_data, (-1, 32, 32, 1))
    print('length of raw data ' + str(len(raw_data)))
    print('raw data shape ', raw_data.shape)
    for i in range(0, len(raw_data), n_properties): 
        concat_array = raw_data[i]

        for j in range(1, n_properties): 
            array2 = raw_data[i + j]
            concat_array = np.concatenate((concat_array, array2), axis=2)
        
        concat_array = np.reshape(concat_array, (32, 32, n_properties))
        input_shape = concat_array.shape
        final_data.append(concat_array)

    final_data = np.reshape(final_data, (-1, 32, 32, n_properties))
    print('length of processed data ' + str(len(final_data)))

    final_data.astype('float32')/255.0 # normalize values to be between 0 and 1

    return final_data, input_shape


def plot_property_distribution(data, anomaly_data, properties):
    for i in range(len(properties)):
        prop = properties[i]
        data_n = data[:, :, :, i].reshape(-1)
        anomaly_data_n = anomaly_data[:, :, :, i].reshape(-1)
        data_n = data_n[data_n != 0]
        anomaly_data_n = anomaly_data_n[anomaly_data_n != 0]

        bins = 50  # Number of bins
        bin_range = (min(np.min(data_n), np.min(anomaly_data_n)),
                     max(np.max(data_n), np.max(anomaly_data_n)))

        plt.hist(data_n, density=True, label='Non Anomalous', alpha=0.5, bins=bins, range=bin_range)
        plt.hist(anomaly_data_n, density=True, label='Anomalous', alpha=0.5, bins=bins, range=bin_range)
        plt.legend()
        plt.xlabel(str(prop))
        plt.ylabel('Fraction of Events')
        plt.title('Distribution of ' + str(prop))
        plt.savefig('figures_d0_dz/Distribution of ' + str(prop))
        plt.clf()



class Autoencoder(): 
    def __init__(self, input_shape, units): 
        self.input_shape = input_shape
        self.pix = input_shape[0]
        self.n_properties = input_shape[2]
        self.hp_units = units
    
    def build_model(self):  
        alpha_init = np.random.randn()
        input_layer = Input(shape=input_shape)
        x = Conv2D(10, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(input_layer)
        x = Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(x)
        x = Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(x)
        x = Flatten()(x)
        encoded = Dense(self.hp_units, activation=lambda x: relu(x, alpha=alpha_init))(x)

        # Decoder layers
        x = Dense(100, activation=lambda x: relu(x, alpha=alpha_init))(encoded)
        x = Dense(64, activation=lambda x: relu(x, alpha=alpha_init))(encoded)
        x = Reshape((8, 8, 1))(x)
        x = Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init))(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(self.n_properties, kernel_size=(4, 4), padding='same')(x)

        self.model = Model(input_layer, x)
        print('latent space dimension:', self.hp_units)
    
    def train(self, train_data, test_data, epochs): 
        learning_rate = .0001
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                        loss='mse',
                        metrics=['accuracy'])
        num_parameters = self.model.count_params()
        print("number of training events: " + str(train_data.shape[0]))
        print('number of parameters: ' + str(num_parameters))

        early_stopping = EarlyStopping(monitor='val_loss', patience=20) 
        checkpoint_callback = ModelCheckpoint(filepath='model_d0_dz/weights_epoch_{epoch:02d}.h5', save_freq='epoch', period=5)
        self.history = self.model.fit(train_data, train_data, epochs=epochs, batch_size=500, validation_data=(test_data, test_data), callbacks=[early_stopping, checkpoint_callback])

        self.train_loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']

        self.model.save('model_d0_dz/model.h5')

    def plot_loss(self): 
        plt.plot(self.train_loss, label='Training Loss')
        plt.plot(self.val_loss, label='Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.savefig('figures_d0_dz/Training and Validation Losses')
        plt.clf()
    
    def mse(self, test_data, anomaly_data): 
        reconstructed_anomaly = self.model.predict(anomaly_data)
        anomaly_loss = np.mean(np.square(anomaly_data - reconstructed_anomaly), axis=(1,2,3))
    
        # self.anomaly_scores = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(anomaly_data, reconstructed_anomaly).numpy()
        reconstructed_test = self.model.predict(test_data)
        test_loss = np.mean(np.square(test_data - reconstructed_test), axis=(1,2,3))
        # self.test_scores = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(test_data, reconstructed_test).numpy()
        self.anomaly_scores = anomaly_loss
        self.test_scores = test_loss
        print('anomaly MSE (loss) over all anomalous inputs: ', np.mean(anomaly_loss))
        print('not anomaly MSE (loss) over all non-anomalous inputs: ', np.mean(test_loss))

    def plot_anomaly_score_distribution(self):
        plt.figure()
        bins = 50  
        range_ = (min(np.min(self.anomaly_scores), np.min(self.test_scores)),
                max(np.max(self.anomaly_scores), np.max(self.test_scores)))

        weights_anomaly = np.ones_like(self.anomaly_scores.flatten()) / len(self.anomaly_scores.flatten())
        weights_test = np.ones_like(self.test_scores.flatten()) / len(self.test_scores.flatten())

        plt.hist(self.anomaly_scores.flatten(), bins=bins, range=range_, weights=weights_anomaly,
                color='red', alpha=0.5, label='anomalous', density=False)
        plt.hist(self.test_scores.flatten(), bins=bins, range=range_, weights=weights_test,
                color='blue', alpha=0.5, label='not anomalous', density=False)

        plt.xlabel('Anomaly Score')
        plt.title('Anomaly Score Distribution')
        plt.legend()
        plt.savefig('figures_d0_dz/Anomaly Score Distribution')
        plt.clf()



    def plot_roc(self, test_data, anomaly_data):
        all_data = np.concatenate((test_data, anomaly_data), axis=0)
        data_pred = self.model.predict(all_data)
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
        plt.plot(fprs, tprs, label='Autoencoder')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver operating characteristic (ROC) curve')
        plt.legend()
        plt.savefig("figures_d0_dz/ROC")
        plt.clf()

        auc_score = auc(fprs, tprs)
        print('AUC: {:.3f}'.format(auc_score))
    
    def plot_example(self, test_data, properties, title):
        n_epochs = len(self.train_loss)
        n_files = round(n_epochs / 5)
        example_norm = test_data[8].reshape(self.input_shape)

        for j in range(len(properties)): 
            reco = [] 
            squared_err = []

            fig, axs = plt.subplots(n_files + 1, 2, figsize=(15, 5 * n_files + 5))
            max_value = np.max(example_norm[:,:,j])
            min_value = np.min(example_norm[:,:,j])
            axs[0, 0].imshow(example_norm[:,:,j], cmap='viridis', vmax=max_value, vmin=min_value)
            axs[0, 0].set_title('Input ' + title + ' Jet ' + properties[j])
            axs[0, 1].axis('off')

            for i in range(n_files): 
                weight_file = 'model_d0_dz/weights_epoch_{:02d}.h5'.format((i + 1) * 5)
                weight_model = tf.keras.models.load_model(weight_file, custom_objects={'<lambda>': lambda x: relu(x, alpha=0.5)})
                reco_ex = weight_model.predict(example_norm.reshape(1, self.pix, self.pix, self.n_properties)).reshape(self.input_shape)
                reco.append(reco_ex)

                squared_err.append(((example_norm - reco[i])**2).reshape(self.pix, self.pix, self.n_properties)[:, :, 0])

                axs[i + 1, 0].imshow(reco[i][:, :, j].reshape((self.pix, self.pix, 1)), cmap='viridis')
                axs[i + 1, 0].set_title('Reconstructed' + title + ' Jet (Epoch ' + str((i+1)*5) + ')')

                axs[i + 1, 1].imshow(squared_err[i], cmap='viridis')
                axs[i + 1, 1].set_title('Sq2 Error: (X-f(X))^2 (Epoch ' + str((i+1)*5) + ')')

            plt.tight_layout()
            plt.savefig("figures_d0_dz/Example Reconstruction " + properties[j] + ' ' + title)
            plt.show()
            plt.clf()

    def run_all(self, train_data, test_data, anomaly_data, epochs, properties): 
        self.build_model()
        self.train(train_data, test_data, epochs = epochs)
        self.plot_loss()
        self.mse(test_data, anomaly_data)
        self.plot_anomaly_score_distribution()
        self.plot_roc(test_data, anomaly_data)
        self.plot_example(test_data, properties, 'qcd')
        self.plot_example(anomaly_data, properties, 'wjet')

''' IMPORTANT THINGS TO CHANGE '''

epochs = 50

qcd_file = "vector_results_newest/qcd.npy"
wjet_file = 'vector_results_newest/wjet.npy'
properties = ['pT', 'impact parameter'] # must be in order of file 

n = len(properties) # number of properties 

hp_units = 12

''''''''''''''''''''''''''''''''''''

qcd_data, input_shape = reshape_data(qcd_file, n)
train_qcd, test_qcd = train_test_split(qcd_data, test_size=0.5)
wjet_data = reshape_data(wjet_file, n)[0]

plot_property_distribution(qcd_data, wjet_data, properties)
model = Autoencoder(input_shape, hp_units)
model.run_all(train_qcd, test_qcd, wjet_data, epochs, properties)