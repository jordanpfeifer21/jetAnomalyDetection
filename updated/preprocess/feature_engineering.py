import numpy as np
import pandas as pd
from typing import List

def calculate_d_over_dErr(row: pd.Series, label: str, valid_pdg: List[str]) -> np.ndarray:
    if label not in ['dz', 'dxy', 'd0']:
        raise ValueError(f"Invalid label {label}. Choose from 'dz', 'dxy', 'd0'.")
    arr = np.array(row[label]) / np.array(row[(label + 'Err')])
    arr = np.where(np.array(row[(label + 'Err')]) >= 0.0, arr, 0.0)
    arr = np.where(np.isin(np.array(row['pdgId']), valid_pdg), arr, 0.0)
    arr = np.clip(arr, -5.0, 5.0)
    return arr

def calculate_dR(row: pd.Series) -> np.ndarray:
    eta = np.where(np.array(row['puppiWeight']) == 1.0, row['eta'], 0.0)
    return np.sqrt(np.array(row['eta'])**2 + np.array(row['phi'])**2)

def within_bounds(row: pd.Series) -> np.ndarray:
    return np.intersect1d(
        np.where(np.abs(np.array(row['dR'])) <= 0.8)[0],
        np.where(np.array(row['puppiWeight']) == 1.0)[0]
    )

def filter_row(row: pd.Series, indices: np.ndarray) -> pd.Series:
    for col in row.index:
        if col != 'within_bounds':
            row[col] = np.array(row[col])[indices].flatten()
    return row

def one_hot_encode_pdgId(row: pd.Series, pdg_ids: List[int]) -> pd.Series:
    pdg_array = np.array(row['pdgId'])
    for pdg_id in pdg_ids:
        row[f'pdgId_{pdg_id}'] = (pdg_array == pdg_id).astype(int)
    return row

def modify_df(df: pd.DataFrame, pdg: List[str]) -> pd.DataFrame:
    df['dz/dzErr'] = df.apply(lambda row: calculate_d_over_dErr(row, label='dz', valid_pdg=pdg), axis=1)
    df['d0/d0Err'] = df.apply(lambda row: calculate_d_over_dErr(row, label='d0', valid_pdg=pdg), axis=1)
    df['dR'] = df.apply(calculate_dR, axis=1)
    df['within_bounds'] = df.apply(within_bounds, axis=1)
    df['log_pt'] = df.apply(lambda row: np.log(np.array(row['pt'])), axis=1)

    unique_pdg_ids = sorted(df['pdgId'].explode().unique().tolist())
    df = df.apply(lambda row: one_hot_encode_pdgId(row, unique_pdg_ids), axis=1)
    df = df.apply(lambda row: filter_row(row, row['within_bounds']), axis=1)

    return df