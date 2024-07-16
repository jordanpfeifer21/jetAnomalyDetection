import pandas as pd
from data_analysis import make_histogram
import numpy as np
# import dgl #deep graph network that will make integrating graphs easier
import torch
from scipy.spatial.distance import cdist
import constants as c
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from scipy.stats import skew
def one_hot_encode_list(unique_values, array):
    enc = preprocessing.OneHotEncoder(categories = [unique_values], handle_unknown="error")
    one_hot_lists = {value: [] for value in unique_values}
    for arr in array:
        encoded = enc.fit_transform(np.array(arr).reshape(-1,1)).toarray()
        for i, value in enumerate(unique_values):
            one_hot_lists[value].append(encoded[:, i].tolist())
    return one_hot_lists
#make 2D histogram of the data
def format_2D(data, properties, scalers=None):
    hists = []
    n_properties = len(properties)
    if scalers == None:
        scalers = []
    vals = []
    for i, prop in enumerate(properties):
        if prop == 'pdgId':
            flattened_list = [item for sublist in data['pdgId'] for item in sublist]
            unique_values_list = sorted(list(set(flattened_list)))
            print(unique_values_list)
            value_to_index = {value: index + 1 for index, value in enumerate(unique_values_list)}
            print(value_to_index)
            vals.append([value_to_index[item] for item in flattened_list])
            hist_list = []
            for j in range(data.shape[0]):
                d = data['pdgId'][j]
                d = [value_to_index[item] for item in d]
                hist_list.append(make_histogram(data['eta'][j], data['phi'][j], d))
            hists.append(hist_list)
            # one_hot_dict = one_hot_encode_list(unique_values_list, data['pdgId'])
            # particle_ids = []
            # for key in one_hot_dict:
            #     one_hot_list = one_hot_dict[key]
            #     vals.append([item for sublist in one_hot_list for item in sublist])
            #     hists.append([make_histogram(data['eta'][j], data['phi'][j], one_hot_list[j]) for j in range(data.shape[0])])
            #     particle_ids.append(key)
            n_properties += len(hists) - 1
            # print(hists)
        elif prop[-3:] == "Err":
            '''
            currently only makes sense for dz and d0
            '''
            vals.append([])
            prop1 = prop[:2]
            valid_pdg = [-11, 11, -13, 13, -211, 211]
            all_vals = [p / z for sublist1, sublist2, sublist3 in zip(data[prop1], data[prop], data['pdgId']) for p, z, x in zip(sublist1, sublist2, sublist3) if z >=0 and x in valid_pdg and np.abs(p/z) <= 5.0]
            if len(scalers) < i + 1:
                scaler = preprocessing.StandardScaler().fit(np.array(all_vals).reshape(-1,1))
                scalers.append(scaler)
            hist_list = []
            for j in range(data.shape[0]):
                prop_data = pd.to_numeric(data[prop1][j])/pd.to_numeric(data[prop][j])
                valid_indices = np.array(data[prop][j]) >= 0
                # checks that the error is greater than 0 and that it is within 5sigma of the original distribution
                prop_data = np.array(prop_data)[valid_indices]
                within5sigmaindices = np.where(np.abs(prop_data) <= 5.0)
                prop_data = prop_data[within5sigmaindices]
                if len(prop_data) > 0:
                    # prop_scaled = scalers[i].transform(prop_data.reshape(-1,1))
                    prop_scaled = prop_data
                    filtered_eta = np.array(data['eta'][j])[valid_indices][within5sigmaindices].flatten()
                    filtered_phi = np.array(data['phi'][j])[valid_indices][within5sigmaindices].flatten()
                    filtered_prop = np.array(prop_scaled).flatten()
                    vals[i].extend(filtered_prop)
                    hist_list.append(make_histogram(filtered_eta, filtered_phi, filtered_prop))
                else:
                    print('error, length of data is ', len(prop_data))
                    hist_list.append(make_histogram([0], [0], [0]))
            print(np.mean([x * x for x in vals[i]]))
            hists.append(hist_list)
        else:
            vals.append([])
            flattened_list = [item for sublist in data[prop] for item in sublist]
            unique_values_list = sorted(list(set(flattened_list)))
            if len(scalers) < i + 1:
                s = preprocessing.MinMaxScaler(feature_range = (0, 1/(np.std(flattened_list))))
                f = s.fit_transform(np.array(flattened_list).reshape(-1,1))
                scaler = preprocessing.MinMaxScaler(feature_range = (0, 1/(np.std(flattened_list) * np.sqrt(np.mean([x * x for x in f])))))
                scaler.fit(np.array(flattened_list).reshape(-1,1))
                print("DONE")
                scalers.append(scaler)
            if 0.0 in unique_values_list:
                print('warning, 0 in data will cause incorrect data translation to histograms')
            hists.append([make_histogram(data['eta'][j], data['phi'][j], (scalers[i].transform(np.array(data[prop][j]).reshape(-1,1))).flatten()) for j in range(data.shape[0])])
            vals[i].extend(scalers[i].transform(np.array(flattened_list).reshape(-1,1)).flatten())
            
    total_hist = np.stack((hists), axis=-1)
    total_hist = np.reshape(total_hist, (-1, n_properties, c.BINS, c.BINS )).astype('float32')
    print("Length of data: ", len(total_hist))
    return total_hist, scalers, vals
def load_files(background_file, signal_file):
    background = pd.read_pickle(background_file)
    signal = pd.read_pickle(signal_file)
    return background, signal









