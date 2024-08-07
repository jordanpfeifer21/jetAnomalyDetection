import pandas as pd 
from data_analysis import make_histogram
import numpy as np 
# import dgl #deep graph network that will make integrating graphs easier
import torch
torch.cuda.empty_cache()
from scipy.spatial.distance import cdist
import constants as c
import torch.nn as nn 
import torch.nn.functional as F
from sklearn import preprocessing

def one_hot_encode_list(unique_values, array):
    enc = preprocessing.OneHotEncoder(categories = [unique_values], handle_unknown="error")
    one_hot_lists = {value: [] for value in unique_values}

    for arr in array: 
        encoded = enc.fit_transform(np.array(arr).reshape(-1,1)).toarray()
        for i, value in enumerate(unique_values):
            one_hot_lists[value].append(encoded[:, i].tolist())

    return one_hot_lists


#make 2D histogram of the data
def format_2D(data, properties):
    hists = []
    n_properties = len(properties)
    print("Why is this the wrong number: " + str(properties) + "," + str(len(properties)))
    for prop in properties: 
        if prop == 'pdgId': 
            flattened_list = [item for sublist in data['pdgId'] for item in sublist]
            unique_values_list = sorted(list(set(flattened_list)))
            value_to_index = {value: index + 1 for index, value in enumerate(unique_values_list)}
            one_hot_dict = one_hot_encode_list(unique_values_list, data['pdgId'])
            particle_ids = []
            for key in one_hot_dict: 
                one_hot_list = one_hot_dict[key]
                # hists.append([make_histogram(data['eta'][i], data['phi'][i],
                #                     [value_to_index[number] for number in one_hot_list[i]]) for i in range(data.shape[0])])
                hists.append([make_histogram(data['eta'][i], data['phi'][i], one_hot_list[i]) for i in range(data.shape[0])])
                particle_ids.append(key)
            n_properties += len(hists) - 1
        elif prop == 'dzErr': 
            # TODO: fix this so that it is always x/x_err if err exists
            dz = [p / z for sublist1, sublist2 in zip(data['dz'], data['dzErr']) for p, z in zip(sublist1, sublist2) if z >=0]
            dz = np.array(dz)[np.abs(dz) + np.mean(dz) <= 3*np.std(dz)].flatten()
            scaler = preprocessing.StandardScaler(with_mean=False).fit(np.array(dz).reshape(-1,1))
            dz = scaler.transform(np.array(dz).reshape(-1,1))
            hist_list = []
            for i in range(data.shape[0]):
                prop_data = pd.to_numeric(data['dz'][i])/pd.to_numeric(data['dzErr'][i])
                valid_indices = np.array(data['dzErr'][i]) >= 0 
                prop_data = np.array(prop_data)[valid_indices]

                try: 
                    dz_scaled = scaler.transform(prop_data.reshape(-1,1))
                    within_3sigma_indices = (np.abs(dz_scaled) <= 3).flatten()
                    filtered_eta = np.array(data['eta'][i])[valid_indices][within_3sigma_indices].flatten()
                    filtered_phi = np.array(data['phi'][i])[valid_indices][within_3sigma_indices].flatten()
                    filtered_dz = np.array(dz_scaled)[within_3sigma_indices].flatten()
                    hist_list.append(make_histogram(filtered_eta, filtered_phi, filtered_dz))
                except: 
                    dz_scaled = np.zeros(prop_data)
                    print('error')
                    hist_list.append(make_histogram([0], [0], [0]))
                
            hists.append(hist_list)
            # hists.append([make_histogram(data['eta'][i], data['phi'][i], pd.to_numeric(data['dz'][i])/pd.to_numeric(data['dzErr'][i])) for i in range(data.shape[0])])
        else: 
            flattened_list = [item for sublist in data[prop] for item in sublist]
            # scaler = preprocessing.StandardScaler().fit(np.array(flattened_list).reshape(-1,1))
            unique_values_list = sorted(list(set(flattened_list)))
            # print("max value " + prop + ': ' + str(np.max(unique_values_list)))
            # print("min value " + prop + ': ' + str(np.min(unique_values_list)))

            if 0.0 in unique_values_list: 
                print('warning, 0 in data will cause incorrect data translation to histograms')
            hists.append([make_histogram(data['eta'][i], data['phi'][i], (np.array(data[prop][i]) * 10).flatten()) for i in range(data.shape[0])])
            # print("warning: multiplying by 10")
    total_hist = np.stack((hists), axis=-1)
    print("Number of properties: " + str(n_properties))
    total_hist = np.reshape(total_hist, (-1, n_properties, c.BINS, c.BINS)).astype('float32')

    print("Length of data: ", len(total_hist))
    return total_hist



def load_files(background_file, signal_file):
    background = pd.read_pickle(background_file)
    signal = pd.read_pickle(signal_file) 
    return background, signal 

