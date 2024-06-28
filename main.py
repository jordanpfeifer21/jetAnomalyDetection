

from models import Autoencoder, VariableAutoencoder # import the autoencoder from the models file
import tensorflow as tf # import tensorflow
import numpy as np  # import numpy
from format_data import format_2D, load_files, get_formatted_shape #import functions from format_data file
from model_analysis import plot_anomaly_score, loss, roc #import functions to make plots from model_analysis file
from sklearn.model_selection import train_test_split #import
from keras.callbacks import EarlyStopping, ModelCheckpoint #import early stopping and model checkpoint


#the function that trains the model
def train(model, train_data, test_data):
    """ Training routine. """
    learning_rate = .0001 #define the learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) #define the optimizer functions
    model.compile(optimizer=optimizer,
                    loss='mse',
                    metrics=['accuracy']) #configure the model for training (define loss function and metric)
    num_parameters = model.count_params() #find the number of parameters
    print(train_data.shape)
    print("number of training events: " + str(train_data.shape[0]))
    print('number of parameters: ' + str(num_parameters))

    
    def learning_rate_scheduler(epoch, learning_rate):
            if epoch < 5:
                return learning_rate  # Keep the initial learning rate for the first 10 epochs
            else:
                return learning_rate * tf.math.exp(-0.1)  # Decrease the learning rate exponentially

    # Create a callback for the learning rate scheduler
    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10) # is loss has not change din 20 epochs stop the model
    checkpoint_callback = ModelCheckpoint(filepath='model_d0_dz/weights_epoch_{epoch:02d}.tf', save_freq='epoch', period=5) #save the weights of the model with the lowest loss

    #train the model and store the loss values
    model.history = model.fit(train_data, train_data, epochs=epochs, batch_size=500, validation_data=(test_data, test_data), callbacks=[early_stopping, checkpoint_callback, lr_scheduler_callback])



    model.train_loss = model.history.history['loss']
    model.val_loss = model.history.history['val_loss']

    # model.save('model_d0_dz/model.h5')

#find the mse    
def mse(model, test_data, anomaly_data): 
    reconstructed_anomaly = model.predict(anomaly_data) #pass known anomaly data into model
    model.anomaly_scores = np.mean(np.square(anomaly_data - reconstructed_anomaly), axis=(1,2,3)) #compute the anomaly scores

    #pass known background data into the model
    reconstructed_test = model.predict(test_data)
    model.test_scores = np.mean(np.square(test_data - reconstructed_test), axis=(1,2,3))

    #print mse over anomolous and non anomolus data
    print('anomaly MSE (loss) over all anomalous inputs: ', np.mean(model.anomaly_scores)) 
    print('not anomaly MSE (loss) over all non-anomalous inputs: ', np.mean( model.test_scores))

''' IMPORTANT THINGS TO CHANGE '''

epochs = 50

properties = ['pT'] # must be in order of file 

n = len(properties) # number of properties 

hp_units = 12

'''''''''''''''''''''''''''''''''''' 

def reshape_data(file_name, n_properties=2):
    final_data = []
    raw_data = np.load(file_name, allow_pickle=True)
    raw_data = np.reshape(raw_data, (-1, 32, 32, 2))
    print('length of raw data ' + str(len(raw_data)))
    for i in range(0, len(raw_data)):
        input_shape = raw_data[i].shape
        final_data.append(raw_data[i])
    final_data = np.reshape(final_data, (-1, 32, 32, n_properties))
    print('length of processed data ' + str(len(final_data)))
    final_data.astype('float32')/255.0 # normalize values to be between 0 and 1
    return final_data, input_shape
qcd_file = '/isilon/export/home/rpankaj/jetAD/data/Ekin/output_array-qcd-61440-dz-pT.npy'
background, input_shape = reshape_data(qcd_file)

train_qcd, test_qcd = train_test_split(background, test_size=0.5)
signal = reshape_data('/isilon/export/home/rpankaj/jetAD/data/Ekin/output_array-HtoBB-61440-dz-pT.npy')[0]

'''
#define data files
background_file = '/isilon/export/home/rpankaj/jetAD/data/processed_data/background.pkl'
signal_file = '/isilon/export/home/rpankaj/jetAD/data/processed_data/signal.pkl'
background, signal = load_files(background_file, signal_file)
background = format_2D(background).astype('float32')/255.0


#make a 2D histogram of the data
signal = format_2D(signal).astype('float32')/255.0

input_shape = get_formatted_shape(background)

train_qcd, test_qcd = train_test_split(background, test_size=0.5) #separate the data 50/50 between testing and trainging
'''

print("---------------------------------")
print(train_qcd.shape)
print("+++++++++++++++++++++++++++++++")


model = VariableAutoencoder(input_shape) #make an instange of the model obect
model(tf.keras.Input(shape=input_shape))

model.summary()

train(model, train_qcd, test_qcd) #train the model
loss(model) #plot the loss for the model
mse(model, test_qcd, signal) #calculate the mean squared error
plot_anomaly_score(model)#plot anonmaly score
roc(model, test_qcd, signal)

