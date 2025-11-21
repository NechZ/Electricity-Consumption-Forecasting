import pandas as pd
import numpy as np
from typing import Dict, List

from .preprocessing import TimeseriesDataSet

def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs((actual - predicted)))

def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.sqrt(np.mean((actual - predicted) ** 2))

def align_ground_truth(full_series: np.ndarray, input_size: int, output_size: int) -> np.ndarray:
    """
    Extracts the ground truth values corresponding exactly to the tiled windows
    predicted by the model.
    
    Logic:
    The model predicts at indices: range(input_size, len + 1, output_size).
    For a split point `i`, the ground truth is series[i : i + output_size].
    We must discard any window where we don't have the full output_size truth available
    (to match the safe prediction logic).
    """
    ground_truth_segments = []
    
    # This range must match the 'valid_split_points' in training.py -> predict()
    valid_split_points = range(input_size, len(full_series) + 1, output_size)
    
    for i in valid_split_points:
        # The ground truth for this prediction window
        target_window = full_series[i : i + output_size]
        
        # Only keep it if we have a full window (matching prediction logic)
        if len(target_window) == output_size:
            ground_truth_segments.append(target_window)
            
    if not ground_truth_segments:
        return np.array([])
        
    # Flatten to match the flattened predictions
    return np.concatenate(ground_truth_segments).flatten()

def evaluate_series(
    dataset: TimeseriesDataSet,
    preds: Dict[str, Dict[str, np.ndarray]],
    input_size: int,
    output_size: int
) -> Dict[str, Dict[str, float]]:
    """
    Calculates metrics by aligning the raw series data to the strided predictions.
    """
    summary = {}
    
    for key, pred_dict in preds.items():
        # Get the full original unscaled series
        full_unscaled = dataset.get_unscaled_data(key)
        train_full = full_unscaled["train"]
        val_full = full_unscaled["validation"]
        
        # Align ground truth to predictions
        train_true = align_ground_truth(train_full, input_size, output_size)
        val_true   = align_ground_truth(val_full,   input_size, output_size)
        
        train_pred = pred_dict["train"]
        val_pred   = pred_dict["validation"]

        # Safety check for length mismatches (should be fixed by align_ground_truth)
        min_len_train = min(len(train_true), len(train_pred))
        min_len_val   = min(len(val_true), len(val_pred))
        
        summary[key] = {
            "train_mae": mean_absolute_error(train_true[:min_len_train], train_pred[:min_len_train]),
            "val_mae":   mean_absolute_error(val_true[:min_len_val],     val_pred[:min_len_val]),
            "train_rmse": root_mean_squared_error(train_true[:min_len_train], train_pred[:min_len_train]),
            "val_rmse":   root_mean_squared_error(val_true[:min_len_val],     val_pred[:min_len_val])
        }
    return summary

def top_n_by_metric(
    summary: Dict[str, Dict[str, float]],
    metric: str,
    n: int = 5,
    reverse: bool = False
) -> Dict[str, float]:
    sorted_items = sorted(
        summary.items(),
        key=lambda kv: (float(kv[1].get(metric, np.inf))),
        reverse=reverse
    )
    return {k: v for k, v in sorted_items[:n]}

def get_full_results_dict(
    dataset: TimeseriesDataSet,
    predictions: Dict[str, Dict[str, np.ndarray]],
    input_size: int,
    output_size: int
) -> Dict[str, pd.DataFrame]:
    """
    Reconstructs a DataFrame aligning timestamps, actuals, and predictions.
    Handles strided/tiled predictions correctly.
    """
    full_results_dict = {}

    for key, series_preds in predictions.items():
        # 1. Build the base DataFrame with all Actuals
        train_true_full = dataset.get_unscaled_data(key)['train']
        val_true_full   = dataset.get_unscaled_data(key)['validation']
        actual_concat   = np.concatenate([train_true_full, val_true_full]).flatten()
        
        df = pd.DataFrame({
            'timestamp': dataset.get_resampled_data()[dataset.timestamp_column],
            'actual': np.nan, # Fill matching length first
            'train_predicted': np.nan,
            'validation_predicted': np.nan
        })
        # Clip df to match actual data length
        df = df.iloc[:len(actual_concat)].copy()
        df['actual'] = actual_concat

        # 2. Fill Training Predictions
        # Logic: iterate valid split points, fill the 'output_size' block
        train_pred_flat = series_preds['train']
        # We must reconstruct the indices to place the flattened array back into time
        current_pred_idx = 0
        valid_split_points_train = range(input_size, len(train_true_full) + 1, output_size)
        
        for i in valid_split_points_train:
            # Check if we have data for this block
            if current_pred_idx + output_size <= len(train_pred_flat):
                # Define the block in the dataframe
                start_idx = i
                end_idx = i + output_size
                
                # Define the block in the flattened prediction array
                p_start = current_pred_idx
                p_end = current_pred_idx + output_size
                
                # Assign
                # Note: ensure indices fit in dataframe
                if end_idx <= len(df):
                    df.iloc[start_idx:end_idx, df.columns.get_loc('train_predicted')] = train_pred_flat[p_start:p_end]
                
                current_pred_idx += output_size

        # 3. Fill Validation Predictions
        val_pred_flat = series_preds['validation']
        current_pred_idx = 0
        offset = len(train_true_full) # Validation starts after train
        
        valid_split_points_val = range(input_size, len(val_true_full) + 1, output_size)
        
        for i in valid_split_points_val:
            if current_pred_idx + output_size <= len(val_pred_flat):
                # Map local validation index 'i' to global dataframe index
                global_start = offset + i
                global_end   = offset + i + output_size
                
                p_start = current_pred_idx
                p_end = current_pred_idx + output_size
                
                if global_end <= len(df):
                    df.iloc[global_start:global_end, df.columns.get_loc('validation_predicted')] = val_pred_flat[p_start:p_end]
                
                current_pred_idx += output_size

        full_results_dict[key] = df

    return full_results_dict