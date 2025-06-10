Jet Anomaly Detection is a PyTorch Geometric-based framework for detecting anomalous high-energy jets using graph neural networks. The system supports both unsupervised (autoencoder-based) and supervised (classifier-based) learning, utilizing particle-level features and k-nearest-neighbor graph constructions.

## Features

- Graph-based autoencoder for unsupervised anomaly detection
- Binary classifier for supervised learning tasks
- Hyperparameter sweep module for model tuning
- Visualization tools for loss curves, ROC curves, anomaly scores
- Modular preprocessing pipeline with feature engineering and normalization

## Usage

- Find the raw root files 
- Preprocess each of the root files at a particular HT range. Use the command line arguments. Run this via: 
```bash
python scripts/preprocessing.py
```
- The output of these files are .pkl files.We save intermediate files in order to minimize the time/compute spent processing root files. 
- Process the .pkl intermediate files (scale, feature engineer, etc.) by running: 
```bash
python scripts/processing.py
```
- Once you have the proper distributions, you can trian an autoencoder 
- Process the .pkl intermediate files (scale, feature engineer, etc.) by running: 
```bash
python scripts/run_train_autoencoder.py
```
or train a classifer 
```bash
python scripts/run_train_classifier.py
```
- Parameter sweeps are performed via 
```bash
python scripts/parameter_sweep.py
```