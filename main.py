from models import Autoencoder 
import tensorflow as tf 
import numpy as np 
from format_data import format_2D, load_files
from model_analysis import plot_anomaly_score, loss, roc
from sklearn.model_selection import train_test_split


def train(model, train_data, test_data):
    """ Training routine. """
    learning_rate = .0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                    loss='mse',
                    metrics=['accuracy'])
    num_parameters = model.count_params()
    print("number of training events: " + str(train_data.shape[0]))
    print('number of parameters: ' + str(num_parameters))

    # early_stopping = EarlyStopping(monitor='val_loss', patience=20) 
    # checkpoint_callback = ModelCheckpoint(filepath='model_d0_dz/weights_epoch_{epoch:02d}.h5', save_freq='epoch', period=5)
    model.history = model.fit(train_data, train_data, epochs=epochs, batch_size=500, validation_data=(test_data, test_data), callbacks=[])

    model.train_loss = model.history.history['loss']
    model.val_loss = model.history.history['val_loss']

    # model.save('model_d0_dz/model.h5')

    
def mse(model, test_data, anomaly_data): 
    reconstructed_anomaly = model.predict(anomaly_data)
    model.anomaly_scores = np.mean(np.square(anomaly_data - reconstructed_anomaly), axis=(1,2,3))

    reconstructed_test = model.predict(test_data)
    model.test_scores = np.mean(np.square(test_data - reconstructed_test), axis=(1,2,3))

    print('anomaly MSE (loss) over all anomalous inputs: ', np.mean(model.anomaly_scores))
    print('not anomaly MSE (loss) over all non-anomalous inputs: ', np.mean( model.test_scores))

''' IMPORTANT THINGS TO CHANGE '''

epochs = 50

properties = ['pT'] # must be in order of file 

n = len(properties) # number of properties 

hp_units = 12

'''''''''''''''''''''''''''''''''''' 

background_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/background.pkl'
signal_file = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data/signal.pkl'
background, signal = load_files(background_file, signal_file)
background = format_2D(background).astype('float32')/255.0
signal = format_2D(signal).astype('float32')/255.0

input_shape = background[0].shape
train_qcd, test_qcd = train_test_split(background, test_size=0.5)

model = Autoencoder()
model(tf.keras.Input(shape=input_shape))
model.summary()
train(model, train_qcd, test_qcd)
loss(model) 
mse(model, test_qcd, signal) 
plot_anomaly_score(model)
roc(model, test_qcd, signal)

