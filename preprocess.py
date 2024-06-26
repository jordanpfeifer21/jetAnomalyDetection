from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import numpy as np
import awkward as ak
from fast_histogram import histogram2d
import os
import pandas as pd 
from tqdm import tqdm 
import warnings
import argparse
import constants as c 

def get_fatjets(events): 
    fatjets = events.FatJet
    store_fj = [fj for fj in fatjets[0]]

    # accept jets that do not have electrons or muons nearby   
    electrons = events.Electron
    electrons = fatjets.nearest(electrons[electrons.pt > c.ELECTRON_PT_LOWER_BOUND])
    muons = events.Muon
    muons = fatjets.nearest(muons[muons.pt > c.MUON_PT_LOWER_BOUND])
    
    mask = (
        (ak.fill_none(fatjets.delta_r(electrons) > c.ELECTRON_R_LOWER_BOUND, True)) &
        (ak.fill_none(fatjets.delta_r(muons) > c.MUON_R_LOWER_BOUND, True)) & 
        (~ak.is_none(fatjets.matched_gen, axis=1)) & 
        (fatjets.delta_r(fatjets.matched_gen) < c.MATCHED_GEN_R_LOWER_BOUND)& 
        (fatjets.pt > c.FATJET_PT_LOWER_BOUND) &
        (abs(fatjets.eta) < c.FATJET_ETA_BOUNDS) 
    )

    fatjets = fatjets[mask]
    fatjets = fatjets[ak.num(fatjets) > 0]
    
    if len(fatjets) == 0: 
        return -1, -1
    if (len(ak.argsort(fatjets.pt, axis=1)[0])) == 0: 
      return -1, -1
    
    fatjets = ak.firsts(fatjets[ak.argsort(fatjets.pt, axis=1)])

    for j, fj in enumerate(store_fj):
      if (abs(fj.eta - fatjets.eta[0]) < c.FATJET_DELTA_ETA_BOUND and
          abs(fj.phi - fatjets.phi[0]) < c.FATJET_DELTA_PHI_BOUND and
          abs(fj.pt - fatjets.pt[0]) < c.FATJET_DELTA_PT_BOUND):
          pfcs = [pfcand["pFCandsIdx"] for pfcand in events.FatJetPFCands[0] if pfcand["jetIdx"] == j]

    return fatjets, pfcs

def process_event_root(events):
    fatjets, pfcs = get_fatjets(events)
    
    if isinstance(fatjets, int): 
        return -1, -1
    pfcands = events.PFCands
    eta = [ak.to_numpy(pfcands['phi'] - fatjets['phi']).flatten()[i] for i in pfcs]
    phi = [ak.to_numpy(pfcands['eta'] - fatjets['eta']).flatten()[i] for i in pfcs]
    pt = [ak.to_numpy(pfcands['pt']/fatjets['pt']).flatten()[i] for i in pfcs]
    # TODO: old ratio based on whether it is qcd or wjet -> this is not model agnostic !!!
      # check that current pt scheme is correct

    properties = [pt, eta, phi]
    property_names = ["pt", "eta", "phi"]

    # add all other fields
    fields = pfcands.fields
    for field in fields: 
        if field != "pt" and field != "eta" and field != "phi": 
            properties.append([ak.to_numpy(pfcands[field]).flatten()[i] for i in pfcs])
            property_names.append(field)

    return properties, property_names

def load_root(filepath):
    data = None
    print(filepath)
    if os.path.splitext((filepath))[-1] == ".root" and os.path.isfile(filepath):
        events = NanoEventsFactory.from_root(filepath, schemaclass = PFNanoAODSchema).events()

    for i in tqdm(range(len(events))):
        properties, property_names = process_event_root(events[i:i+1])
        if properties != -1 and data == None: 
            data = {property_name: [] for property_name in property_names}
            for i, prop in enumerate(properties): 
                data[property_names[i]].append(prop)
        elif properties != -1: 
            for i, prop in enumerate(properties): 
                data[property_names[i]].append(prop)
            if len(data['pt']) == 20: 
                print("20 events reached")
                return pd.DataFrame.from_dict(data)

    return pd.DataFrame.from_dict(data)

def load_h5():
    return

def main(datapath, savepath, datatype, filetype):
    # TODO: should we actually care about this warning? 
    warnings.filterwarnings("ignore", message="Found duplicate branch")
    if filetype == '.h5':
        load_h5()
    else:
        df = load_root(datapath)

    df.to_pickle(savepath + "/" + datatype + ".pkl")

    return

if "__main__": 
    parser = argparse.ArgumentParser(
        prog='Preprocess',
        description='preprocesses jet data for anomaly detection'
    )
    parser.add_argument('--data_path', type=str, required=True, help='path where the data is stored')
    parser.add_argument('--save_path', type=str, required=True, help='path where the processed data will be saved')
    parser.add_argument('--data_type', choices=['background', 'signal'], required=True, help='"background" or "signal"')
    parser.add_argument('--file_type', choices=['.root', '.h5'], required=False, default='.root')

    args = parser.parse_args()
    data_path = args.data_path
    save_path = args.save_path
    data_type = args.data_type
    file_type = args.file_type

    main(data_path, save_path, data_type, file_type)


# example call: 
# python preprocess.py --data_path '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/data/WJET/400to600/nano_mc2018_1-1.root' --save_path '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data' --data_type 'signal' --file_type '.root'
    
# background ='/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/data/QCD/300to500/nano_mc2018_12_a677915bd61e6c9ff968b87c36658d9d_0.root'
# signal = '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/data/WJET/400to600/nano_mc2018_1-1.root'
# main(signal, '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data', 'signal', '.root')
