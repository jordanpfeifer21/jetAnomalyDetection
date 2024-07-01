from format_data import format_2D, reshape_data, load_files, get_formatted_shape
from models import Autoencoder, train_model, test_loop
from loss_functions import mse
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model_analysis import mse, plot_anomaly_score_distribution, plot_roc
import numpy as np
#================ Set Up data =======


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

train_data, test_data = train_test_split(background, test_size=0.5)
signal = reshape_data('/isilon/export/home/rpankaj/jetAD/data/Ekin/output_array-HtoBB-61440-dz-pT.npy')[0]

'''
background_file = "/isilon/export/home/rpankaj/jetAD/data/processed_data/background.pkl"
signal_file = "/isilon/export/home/rpankaj/jetAD/data/processed_data/signal.pkl"

background, signal = load_files(background_file, signal_file)
background = format_2D(background).astype('float32')/255.0


#make a 2D histogram of the data
signal = format_2D(signal).astype('float32')/255.0

#signal = reshape_data(signal_file)

train_data, test_data = train_test_split(background, test_size = 0.5)
'''


input_shape = train_data.shape
batch_size = len(train_data)
#=============== Define Model =========
epochs = 10
initial_lr = 0.001

model = Autoencoder(input_shape)

model_name = "Autoencoder pt with 2d Data"

criterion = MSELoss()
optimizer = Adam(model.parameters(), lr = initial_lr)
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print(type(train_data))
print(type(test_data))
print(type(signal))
print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
#=============== Run Model =========
train_loss, test_loss = train_model(
    dataloader_train= train_data, 
    dataLoader_test = test_data, 
    model = model, 
    loss_func = MSELoss(), 
    optimizer = optimizer, 
    epochs = epochs, 
    batch_size = batch_size)
#summary(model, input_size = (32,32,1),batch_size =  10)
print("--------------------------------")
print("Train Loss: ", train_loss)
print("Test Loss: ", test_loss)


#============== Save Analysis========
print(1)
plt.plot(train_loss, label="Training Loss")
plt.plot(test_loss, label ="Validation Loss")
print(2)
plt.legend()
print(3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
print(4)
plt.title('Training and Validation Losses')
plt.savefig('Training and Validation Losses')
print(5)
plt.clf()

mse(model, test_data, signal)
plot_anomaly_score_distribution(model)
plot_roc(model, test_data, signal)

