import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import math
from tqdm import tqdm

def find_scalers(df: pd.DataFrame, df_label: str, cols: List[str]) -> Dict[str, np.ndarray]:
    scaler_dict = {}
    for col in cols:
        if col[:3] != 'pdg':
            flattened_list = sorted(df[col].explode())
            indices = [i for i, item in tqdm(enumerate(flattened_list), desc=f"Finding scalers for {df_label} - {col}") if item != 0.0 and not math.isnan(item)]
            percentiles = np.percentile(np.array(flattened_list)[indices].flatten(), [16, 84])
            scaler_dict[col] = percentiles
        else:
            scaler_dict[col] = [-1]
    return scaler_dict

def apply_scalers(df: pd.DataFrame, scaler_dict: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, dict, dict, dict]:
    data_dict = {}
    org_data_dict = {}
    scaled_zero = {}

    for col in df.columns:
        if col in scaler_dict.keys():
            print(f'Standardizing {col}')
            flattened_list = df[col].explode()
            indices = [i for i, item in enumerate(flattened_list) if item != 0.0]
            org_data_dict[col] = flattened_list

            if col[:3] != 'pdg':
                per_minus_36, per_plus_36 = scaler_dict[col]
                df[col] = df.apply(
                    lambda row: (((np.array(row[col]).reshape(-1, 1)) - per_minus_36) /
                                 (per_plus_36 - per_minus_36) * 2 - 1).flatten(), axis=1
                )
                zero = (0.0 - per_minus_36) / (per_plus_36 - per_minus_36) * 2 - 1
                scaled_zero[col] = zero
                data_dict[col] = [item for sublist in df[col] for item in sublist]
            else:
                data_dict[col] = [item for sublist in df[col] for item in np.array(sublist).flatten()]
                scaled_zero[col] = np.nan

    return df, data_dict, org_data_dict, scaled_zero