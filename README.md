
## About:
The code in this branch was developed to test using a variable autoencoder to detect anomalies in jet data. 

## Explintion of workflow:

Both main.py and models.py contain code for classical uses of an autoencoder and variable autoencoder. In these files variable autoencoders 
are set up to train on a mix of reconstruction loss and kl divergence. Anomaly scores are calculated with the loss function

mainVAE.py and modelVAE.py contain code for using kl_loss as the anomaly scores. The VAE in this code is trained with the same combination of reconstruction and kl_loss.
However anomaly scores are generated only from kl_loss

## Explination of files: 
[constants.py](constants.py): set of constants used throughout the project. Includes parameters for preprosessing and making histograms.

[data_analysis.py](data_analysis.py): is used to plot priperty distributions of the data

[format_data.py](format_data.py): formats data into histograms from raw .pkl files

[main.py](main.py): runs models in models.py

[mainVAE.py](mainVAE.py): runs models in modelsVAE.py

[model_analysis.py](model_analysis.py): for the models in models.py it calculates the AUC and generates the roc curve, trainngs and validation loss plot, and the anomaly score distribution plot. Can also generate recon images

[model_analysisVAE.py](model_analysisVAE.py): for the models in modelsVAE.py it calculates the AUC and generates the roc curve, trainngs and validation loss plot, and the anomaly score distribution plot. Can also generate recon images

[models.py](models.py): contains models that use loss function for anomaly score

[modelsVAE.py](modelsVAE.py): containts models that use kl_loss for anomaly score

[preprocess.py](preprocess.py) and [preprocess.sh](preprocess.sh): process .root files. information on using this code is later in the readme

## Preprocessing data: 

All data (from .root files) must be preprocessed in preprocess.py 
    (data analysis figures are in data_analysis.py, which let us explore variable distributions and data structures)

That data then must then be reformatted and run through a model in main.py 
    (reformatting code is in format_data.py, model analysis figures can be modified in model_analysis.py, 
    model is kept in models.py)
    
## General Information: 
Images are ged into the model as shape (number of parameters, 32, 32). For example if you are training on pt the number of parameters is 1, if on dz/dzErr and d0/d0Err the nyumber of parameters is 2. 
Input shapes for pdgID are of shape (8,32,32) if one-hot-encoded

## Lessons Learned:
 1. You cannot train a VAE on only kl_loss. The purpose of KLD is to compare two distributions so you have to train on recon loss and then can use kl_loss for anomaly scores during the eval loop
 2. trainings VAEs takes longer than regular autoencoders
 3. MSE loss generally works better than Cross Entropy Loss
 4. The best all around latent space we have found is 6x6, but in some cases a 4x4 or 8x8 has shown better performance

If you have any further questions please feel free to reach out to me at rohan_pankaj[at]brown.edu or rohansp01[at]gmail.com 
