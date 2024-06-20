import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Reshape, UpSampling2D, AveragePooling2D, Conv2DTranspose, Dropout
from tensorflow.keras.activations import relu
import numpy as np 

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.optimizer = tf.keras.optimizers.SGD()
        alpha_init = np.random.randn()
        self.history = None
        self.train_loss = None 
        self.val_loss = None
        self.anomaly_scores = None 
        self.test_scores = None
        self.hp_units = 12
       
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

