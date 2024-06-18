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
import pandas as pd
import pickle

# parser = argparse.ArgumentParser(
#     prog='Preprocess',
#     description='preprocesses jet data for anomaly detection'
# )
# parser.add_argument('--data_path', type=str, required=True, help='path where the data is stored')
# parser.add_argument('--save_path', type=str, required=True, help='path where the processed data will be saved')
# parser.add_argument('--data_type', choices=['background', 'signal'], required=True, help='"background" or "signal"')
# parser.add_argument('--file_type', choices=['.root', '.h5'], required=False, default='.root')

# args = parser.parse_args()
# data_path = args.data_path
# save_path = args.save_path
# data_type = args.data_type
# file_type = args.file_type

# main(data_path, save_path, data_type, file_type)

def get_fatjets(event):
    fatjet = event.FatJet
    store_fj = [fj for fj in fatjet[0]]
    print(fatjet)

    # accept jets that do not have electrons or muons nearby
    electrons = event.Electron
    electrons = fatjet.nearest(electrons[electrons.pt > 20.0])
    muons = event.Muon
    muons = fatjet.nearest(muons[muons.pt > 20.0])

    # add eta and pt cutoffs
    filter_mask = (
        (fatjet.delta_r(electrons) > 0.4) &
        # (fatjet.delta_r(muons) > 0.4) &
        # (fatjet.delta_r(fatjet.matched_gen) < 0.4) &
        # (fatjet.pt > 200.0) &
        # (abs(fatjet.eta) < 2.0) &
        (ak.num(fatjet) > 0)
   )
    fatjet = fatjet[filter_mask]
    print(fatjet)

    # require matched generation particle close to the jet
    fatjet = fatjet[~ak.is_none(fatjet.matched_gen, axis=1)]
    fatjet = fatjet[~ak.is_none(fatjet)]

    if (len(ak.argsort(fatjet.pt, axis=1)[0])) == 0: 
      return -1, -1

    fatjet = ak.firsts(fatjet[ak.argsort(fatjet.pt, axis=1)])
    return fatjet, store_fj

def get_pfcands(event, fatjet):
  # find particle flow candidates
  pfcands = event.PFCands[
    {
        'delta_phi': event.PFCands.delta_phi(fatjet),
        'delta_eta': event.PFCands['eta'] - fatjet['eta'],
        'delta_r': event.PFCands.delta_r(fatjet)
    }
  ]
  return pfcands

def histogram(eta, phi, prop, ids):
  eta_min = eta_max = -0.8, 0.8
  phi_min, phi_max = -0.8, 0.8
  incr = 0.05

  hist_range = [[eta_min, eta_max], [phi_min, phi_max]]
  eta_bins = np.arange(eta_min, eta_max, incr)
  phi_bins = np.arange(phi_min, phi_max, incr)
  image_shape = (eta_bins.shape[0], phi_bins.shape[0])

  return histogram2d(phi, eta, range=hist_range, bins=image_shape, weights=prop)

def process_event_root(events):
  fatjets, store_fj = get_fatjets(events)
  if isinstance(fatjets, int): 
    return -1, -1, -1
  else:
    print('matching FatJet!!!')
    if isinstance(events.PFCands, ak.Array):
        print(events.PFCands.type)  # Print the type of PFCands (e.g., RegularArray, ListArray, etc.)
        if hasattr(events.PFCands, 'fields'):
            print(events.PFCands.fields)  # List all fields if PFCands is a record array or structured array
        elif hasattr(events.PFCands, 'keys'):
            print(events.PFCands.keys())  # Alternatively, use keys() if it's a dict-like structure

  # fatjets istype FatJet: 
    eta = ak.to_numpy(events.PFCands['eta'] - fatjets['eta']).flatten()
    phi = ak.to_numpy(events.PFCands['phi'] - fatjets['phi']).flatten()
    pt = ak.to_numpy(events.PFCands['pt']/fatjets['pt']).flatten()
    # TODO: old ratio based on whether it is qcd or wjet -> this is not model agnostic !!!
      # check that current pt scheme is correct

    for j, fj in enumerate(store_fj):
      if (abs(fj.eta - fatjets.eta[0]) < 0.1 and
          abs(fj.phi - fatjets.phi[0]) < 0.1 and
          abs(fj.pt - fatjets.pt[0]) < 1):
          pfcs = [pfcand["pFCandsIdx"] for pfcand in events.PFCands[0] if pfcand["jetIdx"] == j]

    manual_pt = [pt[i] for i in pfcs]
    manual_eta = [eta[i] for i in pfcs]
    manual_phi = [phi[i] for i in pfcs]

    return manual_pt, manual_eta, manual_phi

  # if fatjets[0] is None: 
  #   print('No matching FatJet')
  #   return -1, -1, -1
  

  return -1, -1, -1

  

def load_root(filepath):
  data = {'pT':[], 'eta':[], 'phi':[]}
  print(filepath)
  if os.path.splitext((filepath))[-1] == ".root" and os.path.isfile(filepath):
    events = NanoEventsFactory.from_root(filepath, schemaclass = PFNanoAODSchema).events()
    # print(len(events))
    for i in range(len(events)):
     pt, eta, phi = process_event_root(events[i:i+1])
     if pt != -1: 
        data['pT'].append(pt)
        data['eta'].append(eta)
        data['phi'].append(phi)

  else:
    pass

  return pd.DataFrame.from_dict(data)

def load_h5():
  return

def main(datapath, savepath, datatype, filetype):
  if filetype == '.h5':
    load_h5()
  else:
    df = load_root(datapath)

  df.to_pickle(savepath + "/" + filetype + ".pkl")

  return

f ='/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/data/QCD/300to500/nano_mc2018_12_a677915bd61e6c9ff968b87c36658d9d_0.root'
main(f, '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data', 'background', '.root')
