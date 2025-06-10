"""
Feature scaling utilities for jet constituent data.

This module standardizes per-feature arrays using percentile-based scaling (16th to 84th percentile)
to minimize sensitivity to outliers. It returns transformed values along with mappings for 
original and scaled data, useful for analysis and visualization.

Functions:
- find_scalers: Computes 16th and 84th percentile values for each feature.
- apply_scalers: Applies scaling to features using computed percentiles.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import math
from tqdm import tqdm

def find_scalers(df: pd.DataFrame, df_label: str, cols: List[str]) -> Dict[str, np.ndarray]:
    """
    Compute scaling values (16th and 84th percentiles) for each feature column.

    Args:
        df (pd.DataFrame): DataFrame with array-valued columns (per-jet particle features).
        df_label (str): Label for progress tracking (e.g., "QCD600to700").
        cols (List[str]): List of column names to compute scalers for.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping feature names to (16th, 84th) percentiles.
                               PDG one-hot columns are skipped and mapped to [-1].
    """
    scaler_dict = {}
    for col in cols:
        if col[:3] != 'pdg':
            flattened_list = sorted(df[col].explode())
            # Keep only valid, non-zero, non-NaN entries
            indices = [i for i, item in tqdm(enumerate(flattened_list),
                                             desc=f"Finding scalers for {df_label} - {col}")
                       if item != 0.0 and not math.isnan(item)]
            percentiles = np.percentile(np.array(flattened_list)[indices].flatten(), [16, 84])
            scaler_dict[col] = percentiles
        else:
            # Skip scaling for PDG one-hot columns
            scaler_dict[col] = [-1]
    return scaler_dict


def apply_scalers(df: pd.DataFrame, scaler_dict: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, dict, dict, dict]:
    """
    Apply standardization to the dataset using precomputed percentile-based scalers.

    Args:
        df (pd.DataFrame): Input DataFrame with array-valued features.
        scaler_dict (Dict[str, np.ndarray]): Output of `find_scalers`.

    Returns:
        Tuple containing:
            - df (pd.DataFrame): Scaled version of input DataFrame.
            - data_dict (dict): Flattened scaled values for each column.
            - org_data_dict (dict): Flattened original (unscaled) values for each column.
            - scaled_zero (dict): Value of 0.0 after scaling, useful for visualizing zero filtering.
    """
    data_dict = {}       # Flattened scaled values
    org_data_dict = {}   # Flattened original values
    scaled_zero = {}     # Value of 0.0 after scaling

    for col in df.columns:
        if col in scaler_dict:
            print(f'Standardizing {col}')
            flattened_list = df[col].explode()
            indices = [i for i, item in enumerate(flattened_list) if item != 0.0]
            org_data_dict[col] = flattened_list

            if col[:3] != 'pdg':
                per_minus_36, per_plus_36 = scaler_dict[col]
                df[col] = df.apply(
                    lambda row: (((np.array(row[col]).reshape(-1, 1)) - per_minus_36) /
                                 (per_plus_36 - per_minus_36) * 2 - 1).flatten(),
                    axis=1
                )
                zero = (0.0 - per_minus_36) / (per_plus_36 - per_minus_36) * 2 - 1
                scaled_zero[col] = zero
                data_dict[col] = [item for sublist in df[col] for item in sublist]
            else:
                # For one-hot columns, preserve raw values
                data_dict[col] = [item for sublist in df[col] for item in np.array(sublist).flatten()]
                scaled_zero[col] = np.nan

    return df, data_dict, org_data_dict, scaled_zero
