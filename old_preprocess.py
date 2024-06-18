from collections import defaultdict
import coffea
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import numpy as np
import awkward as ak
import numba as nb
from fast_histogram import histogram2d
import matplotlib.pyplot as plt
import os
import argparse

def match_manual(fatjets, fatjetpfcands, idx):
    pfcs = []
    for j in range(len(fatjetpfcands[0])):
        if idx == fatjetpfcands[0][j]["jetIdx"]:
            pfcs.append(fatjetpfcands[0][j]["pFCandsIdx"])
    return pfcs, fatjets.eta, fatjets.phi, fatjets.pt

def histo_pfcand(x, y, cc,  hist_range, bins):
    return histogram2d(y, x, range=hist_range, bins=bins, weights=cc)

def plot(x, y, cc, ids):
    eta_min, eta_max = -0.8, 0.8
    phi_min, phi_max = -0.8, 0.8
    incr = 0.05

    hist_range = [[eta_min, eta_max], [phi_min, phi_max]]
    eta_bins = np.arange(eta_min, eta_max, incr)
    phi_bins = np.arange(phi_min, phi_max, incr)
    image_shape = (eta_bins.shape[0], phi_bins.shape[0])

    test = histo_pfcand(x, y, cc, hist_range, image_shape)
    return test

def each_event(events, ids, properties_of_interest, loaded_data, qcd):
    fatjets = events.FatJet
    store_fj = []

    if len(fatjets) == 0: 
        return [-1, -1]

    for i in range(len(fatjets[0])):
        store_fj.append(fatjets[0][i])

    # remove overlapping jet and leptons
    electrons_veto = events.Electron
    electrons_veto = electrons_veto[electrons_veto.pt > 20.0]
    electrons_veto = fatjets.nearest(electrons_veto)
    # accept jet that doesn't have an electron nearby
    electrons_veto_selection = ak.fill_none(fatjets.delta_r(electrons_veto) > 0.4, True)
    fatjets = fatjets[electrons_veto_selection]

    muons_veto = events.Muon
    muons_veto = muons_veto[muons_veto.pt > 20.0]
    muons_veto = fatjets.nearest(muons_veto)
    # accept jet that doesn't have a muon nearby
    muons_veto_selection = ak.fill_none(fatjets.delta_r(muons_veto) > 0.4, True)
    fatjets = fatjets[muons_veto_selection]

    # gen-match
    fatjets = fatjets[~ak.is_none(fatjets.matched_gen, axis=1)]
    fatjets = fatjets[fatjets.delta_r(fatjets.matched_gen) < 0.4]

    # pre-selection
    selections = {}
    selections['pt'] = fatjets.pt > 200.0
    selections['eta'] = abs(fatjets.eta) < 2.0
    selections['all'] = selections['pt'] & selections['eta'] 

    fatjets = fatjets[selections['all']]
    fatjets = fatjets[ak.num(fatjets) > 0]
    fatjets = fatjets[ak.argsort(fatjets.pt, axis=1)]
    fatjets = ak.firsts(fatjets)

    if len(fatjets) != 1:
        return [-1, -1]

    for j in range(len(store_fj)):
        fji = store_fj[j]
        if abs(fji.eta - fatjets.eta[0]) < 0.1 and abs(fji.phi-fatjets.phi[0]) < 0.1 and abs(fji.pt-fatjets.pt[0]) < 1:
            pfcs, feta, fphi, fpt = match_manual(fatjets, events.FatJetPFCands, j)
    if len(pfcs) == 0:
        return [-1, -1]

    fatjetpfcands = events.PFCands
    fatjetpfcands['delta_phi'] = fatjetpfcands.delta_phi(fatjets)
    fatjetpfcands['delta_eta'] = fatjetpfcands['eta'] - fatjets['eta']
    fatjetpfcands['delta_r'] = fatjetpfcands.delta_r(fatjets)

    eta = ak.to_numpy(fatjetpfcands['delta_eta'])
    phi = ak.to_numpy(fatjetpfcands['delta_phi'])

    eta = eta.flatten()
    phi = phi.flatten()

    if qcd[0] == 'q': 
        ratio_value = 1 
    elif qcd[0] == 'w': 
        bin_index = np.digitize(fatjets['pt'], loaded_data['bin_edges']) - 1
        if 0 <= bin_index < len(loaded_data['ratio']):
            ratio_value = loaded_data['ratio'][bin_index]
        else:
            ratio_value = 1  # Number is outside the range of bin edges

    fatjetpfcands['pt_ratio'] = fatjetpfcands['pt']/(fatjets['pt'] * ratio_value)
    manual_pt = []
    manual_eta = []
    manual_phi = []
    manual_sum = []

    pt = ak.to_numpy(fatjetpfcands['pt_ratio'])
    pt = pt.flatten()

    pt_sum = ak.to_numpy(fatjetpfcands['pt'])
    pt_sum = pt_sum.flatten()

    for i in pfcs:
        manual_pt.append(pt[i])
        manual_eta.append(eta[i])
        manual_phi.append(phi[i])
        manual_sum.append(pt_sum[i])

    properties = []

    obj = events.PFCands
    for field in properties_of_interest:
        if field == 'impact': 
            d0 = ak.to_numpy(getattr(obj, 'd0')).flatten()
            d0_err = ak.to_numpy(getattr(obj, 'd0Err')).flatten()
            dz = ak.to_numpy(getattr(obj, 'dz')).flatten()
            dz_err = ak.to_numpy(getattr(obj, 'dzErr')).flatten()
            combined_impact = np.sqrt((d0_err**2 + dz_err**2)/np.sqrt(d0**2 + dz**2))
            try:
                manual = []
                for i in pfcs: 
                    manual.appen(combined_impact[i])
                properties.append(manual)
                print(manual)
                (print(str(field)))
            except: 
                a = 0

        elif field == 'pT': 
            properties.append(manual_pt)

        elif field == 'num_particles': 
            a = 0
            
        else:
            field_name = "events.PFCands." + field
            value = (getattr(obj, field))
            type_val = type(value)
            try: 
                fatjetpfcands[str(field)] = getattr(obj, field)
                manual = []
                arr = ak.to_numpy(getattr(obj, field))
                arr = arr.flatten()
                for i in pfcs: 
                    manual.append(arr[i])
                properties.append(manual)
                print(manual)
                (print(str(field)))
            except: 
                a = [1, 1]

    hist_array = []
    for prop in properties: 
        if prop == 'num_particles': 
            weights = np.ones_like(prop)
            hist_array.append(plot(manual_eta, manual_phi, weights, ids))
        else: 
            hist_array.append(plot(manual_eta, manual_phi, prop, ids))
    n_arrays = len(properties)

    return [hist_array, fatjets['pt']]

def main(file_name, file_type, save_folder, properties):

    loaded_data = np.load('/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/old/ratio_data.npy', allow_pickle=True).item()

    if file_type == 'qcd': 
        name = 'qcd' 
        saving_folder = save_folder + '/QCD/'
    elif file_type == 'wjet': 
        name = 'wjet'
        saving_folder = save_folder + '/WJET/'

    total_events = 0 

    def given_folder(data_folder, filename, total_events):
    
        fname = os.path.join(data_folder, filename)

        if os.path.isfile(fname):
            file_extension = os.path.splitext(fname)[1]

            if file_extension == ".root":
                vector_filename = str(int(0/5000))+'_'+ name + '_' + filename + '.npy'

                if not os.path.isfile(os.path.join(saving_folder, vector_filename)):
                    events = NanoEventsFactory.from_root(fname, schemaclass = PFNanoAODSchema).events()
                    print('EVENTS: ' + str(len(events)))
                    pts = []
                    jet_pt = []

                    for k in range(0, len(events), 5000):
                        for i in range(k, k+5000):
                            out = each_event(events[i:i+1], i, loaded_data = loaded_data, properties_of_interest = properties, qcd = file_type)
                            arrays = out[0]
                            fatjet_pt = out[1]
                            if arrays != -1: 
                                pts.append(arrays)
                                total_events += 1
                                print('total_events: ' + str(total_events))
                                jet_pt.append(fatjet_pt)

                        n_arrays = len(properties)
                        parts = file_name.split('/')
                        vector_filename = str(int(k/5000))+'_' + name + '_' + parts[-1] + str(n_arrays) + '.npy'
                        np.save(os.path.join(saving_folder, vector_filename), pts)
                        np.save(os.path.join(saving_folder, parts[-1] + str(k) + '_jet_pt'), jet_pt)
                else: 
                    print('file already exists')

    parts = file_name.split('/')
    parent_folder = '/'.join(parts[:-1])
    if os.path.isfile(file_name): 
            given_folder(parent_folder, file_name, total_events) 

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default="", help='Filename')
parser.add_argument('--type', type=str, default="")
parser.add_argument('--save_folder', type=str, default="")
parser.add_argument('--properties', nargs='+', type=str, default=[])

args = parser.parse_args()
file_name = args.file
file_type = args.type
save_folder = args.save_folder 
properties = args.properties
main(file_name, file_type, save_folder, properties)