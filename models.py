import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Reshape, UpSampling2D, AveragePooling2D, Conv2DTranspose, Dropout, Input, InputLayer
from tensorflow.keras.activations import relu
import keras
import numpy as np 
import visualkeras
from keras_sequential_ascii import keras2ascii
class Autoencoder(tf.keras.Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()

        #self.input_shape = input_shape

        self.optimizer = tf.keras.optimizers.SGD()
        alpha_init = np.random.randn()
        self.history = None
        self.train_loss = None 
        self.val_loss = None
        self.anomaly_scores = None 
        self.test_scores = None
        self.hp_units = 36
        
       

        self.architecture = [
              Conv2D(10, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)), 
              Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)),
              AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
              Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)),
              Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)),
              Conv2D(5, 8, padding='same', activation='relu'),
              Flatten(),
              Dense(self.hp_units, activation=lambda x: relu(x, alpha=alpha_init)), # latent space? 
              Dense(100, activation=lambda x: relu(x, alpha=alpha_init)),
              Dense(64, activation=lambda x: relu(x, alpha=alpha_init)), 
              Reshape((8, 8, 1)), 
              Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)), 
              UpSampling2D(size=(2, 2)), 
              Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)), 
              UpSampling2D(size=(2, 2)), 
              Conv2DTranspose(1, kernel_size=(4, 4), padding='same')
        ]

    def call(self, x):
        """ Passes input through the network. """
        for layer in self.architecture:
            x = layer(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

class VariableAutoencoder(tf.keras.Model):

    def __init__(self, input_shape):
        super(VariableAutoencoder, self).__init__()

        #self.input_shape = input_shape

        self.optimizer = tf.keras.optimizers.SGD()
        alpha_init = np.random.randn()
        self.history = None
        self.train_loss = None 
        self.val_loss = None
        self.anomaly_scores = None 
        self.test_scores = None
        self.hp_units = 12

        self.encoder = tf.keras.Sequential(
            [
                Conv2D(10, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)), 
                Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)),
                AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)),
                Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)),
                Conv2D(5, 8, padding='same', activation='relu'),
                Flatten(),
                Dense(self.hp_units + self.hp_units)
              
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape = (self.hp_units)),
                Dense(100, activation=lambda x: relu(x, alpha=alpha_init)),
                Dense(64, activation=lambda x: relu(x, alpha=alpha_init)), 
                Reshape((8, 8, 1)), 
                Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)), 
                UpSampling2D(size=(2, 2)), 
                Conv2D(5, kernel_size=(4, 4), padding='same', activation=lambda x: relu(x, alpha=alpha_init)), 
                UpSampling2D(size=(2, 2)), 
                Conv2DTranspose(1, kernel_size=(4, 4), padding='same')
            ]
        )
        #from keras_sequential_ascii import keras2ascii
        #keras2ascii(self.encoder(32,32,1))
        #keras2ascii(self.decoder)
        
    def encode(self,x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits = 2, axis = 1)
        import visualkeras


        #visualkeras.layered_view(self.encoder).show() # display using your system viewer
        visualkeras.layered_view(self.encoder, legend=True, to_file='output.png') # write to disk
        
        from keras_sequential_ascii import keras2ascii
        keras2ascii(self.encoder)
        return mean, logvar
    def reparameterize(self, mean, logvar):
        batch = #tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.keras.backend.random_normal(shape = (batch, dim))
        #eps = tf.random.normal(shape = mean.shape)
        return eps* tf.exp(logvar*0.5) + mean
    def decode(self, z, apply_sigmoid = False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
         #visualkeras.layered_view(self.encoder).show() # display using your system viewer
        visualkeras.layered_view(self.decoder, legend=True, to_file='output.png') # write to disk
        
        #from keras_sequential_ascii import keras2ascii
        keras2ascii(self.decoder)
        return logits
    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        return tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
    def call(self, x):
        """ Passes input through the network. """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x = self.decode(z)
        return x
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)