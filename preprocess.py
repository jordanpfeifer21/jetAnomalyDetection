from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import numpy as np
import awkward as ak
from fast_histogram import histogram2d
import os
import pandas as pd 
from tqdm import tqdm 
import warnings

# def plot(x, y, cc, ids):
#     eta_min, eta_max = -0.8, 0.8
#     phi_min, phi_max = -0.8, 0.8
#     incr = 0.05

#     hist_range = [[eta_min, eta_max], [phi_min, phi_max]]
#     eta_bins = np.arange(eta_min, eta_max, incr)
#     phi_bins = np.arange(phi_min, phi_max, incr)
#     image_shape = (eta_bins.shape[0], phi_bins.shape[0])

#     return histogram2d(y, x, range=hist_range, bins=image_shape, weights=cc)

def get_fatjets(events): 
    fatjets = events.FatJet
    store_fj = [fj for fj in fatjets[0]]

    # accept jets that do not have electrons or muons nearby   
    electrons = events.Electron
    electrons = fatjets.nearest(electrons[electrons.pt > 20.0])
    muons = events.Muon
    muons = fatjets.nearest(muons[muons.pt > 20.0])
    
    mask = (
        (ak.fill_none(fatjets.delta_r(muons) > 0.4, True)) &
        (ak.fill_none(fatjets.delta_r(muons) > 0.4, True)) & 
        (~ak.is_none(fatjets.matched_gen, axis=1)) & 
        (fatjets.delta_r(fatjets.matched_gen) < 0.4)& 
        (fatjets.pt > 200.0) &
        (abs(fatjets.eta) < 2.0) 
    )

    fatjets = fatjets[mask]
    fatjets = fatjets[ak.num(fatjets) > 0]
    
    if len(fatjets) == 0: 
        return -1, -1
    if (len(ak.argsort(fatjets.pt, axis=1)[0])) == 0: 
      return -1, -1
    
    fatjets = ak.firsts(fatjets[ak.argsort(fatjets.pt, axis=1)])

    for j, fj in enumerate(store_fj):
      if (abs(fj.eta - fatjets.eta[0]) < 0.1 and
          abs(fj.phi - fatjets.phi[0]) < 0.1 and
          abs(fj.pt - fatjets.pt[0]) < 1):
          pfcs = [pfcand["pFCandsIdx"] for pfcand in events.FatJetPFCands[0] if pfcand["jetIdx"] == j]

    return fatjets, pfcs

def process_event_root(events):
    fatjets, pfcs = get_fatjets(events)
    
    if isinstance(fatjets, int): 
        return -1, -1, -1 
    
    pfcands = events.PFCands
    eta = [ak.to_numpy(pfcands.delta_phi(fatjets)).flatten()[i] for i in pfcs]
    phi = [ak.to_numpy(pfcands['phi'] - fatjets['phi']).flatten()[i] for i in pfcs]
    pt = [ak.to_numpy(pfcands['pt']/fatjets['pt']).flatten()[i] for i in pfcs]
    # TODO: old ratio based on whether it is qcd or wjet -> this is not model agnostic !!!
      # check that current pt scheme is correct

    # if qcd[0] == 'q': 
    #     ratio_value = 1 
    # elif qcd[0] == 'w': 
    #     bin_index = np.digitize(fatjets['pt'], loaded_data['bin_edges']) - 1
    #     if 0 <= bin_index < len(loaded_data['ratio']):
    #         ratio_value = loaded_data['ratio'][bin_index]
    #     else:
    #         ratio_value = 1  # Number is outside the range of bin edges
    # pfcands['pt_ratio'] = pfcands['pt']/(fatjets['pt'] * ratio_value)
    return pt, eta, phi

def load_root(filepath):
    data = {'pT':[], 'eta':[], 'phi':[]}
    print(filepath)
    if os.path.splitext((filepath))[-1] == ".root" and os.path.isfile(filepath):
        events = NanoEventsFactory.from_root(filepath, schemaclass = PFNanoAODSchema).events()

    for i in tqdm(range(len(events))):
        pt, eta, phi = process_event_root(events[i:i+1])
        if pt != -1: 
            data['pT'].append(pt)
            data['eta'].append(eta)
            data['phi'].append(phi)
        if len(data['pT']) == 100: 
            print("2000 events reached")
            return pd.DataFrame.from_dict(data)
    else:
        pass

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
    print(df)
    df.to_pickle(savepath + "/" + datatype + ".pkl")

    return

f ='/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/data/QCD/300to500/nano_mc2018_12_a677915bd61e6c9ff968b87c36658d9d_0.root'
main(f, '/isilon/data/users/jpfeife2/AutoEncoder-Anomaly-Detection/processed_data', 'background', '.root')

