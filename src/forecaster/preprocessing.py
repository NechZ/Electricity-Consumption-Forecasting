import pandas as pd
import numpy as np
import torch
from typing import Tuple, Type
from bisect import bisect_right
from enum import Enum

from .scalers import Scaler

class Granularity(Enum):
    HOURLY = 'hourly'
    DAILY = 'daily'
    MONTHLY = 'monthly'

    
class SlidingWindowDataset(torch.utils.data.Dataset):
        """
        Generates sliding windows on-the-fly across multiple series to avoid
        pre-materializing all windows in RAM.
        """
        def __init__(self, series_list, cov_array, input_size: int, output_size: int, use_time_covariates: bool):
            self.series_list = series_list                  # list of np.ndarray (scaled), shape [T, 1]
            self.cov_array = cov_array                      # np.ndarray or None, shape [T, C]
            self.input_size = input_size
            self.output_size = output_size
            self.use_time_covariates = use_time_covariates
            self.series_tensors = [torch.tensor(s, dtype=torch.float32) for s in series_list]
            if self.use_time_covariates and self.cov_array is not None:
                self.cov_tensor = torch.tensor(self.cov_array, dtype=torch.float32) # [T, C]
            else:
                self.cov_tensor = None
            # precompute per-series window counts and cumulative offsets
            self.counts = []
            cum = [0]
            for s in self.series_list:
                n = max(0, len(s) - self.input_size - self.output_size + 1)
                self.counts.append(n)
                cum.append(cum[-1] + n)
            self.cum = np.array(cum, dtype=np.int64)  # len = n_series + 1
            self.total = int(self.cum[-1])
    
        def __len__(self):
            return self.total
    
        def __getitem__(self, idx: int):
            si = bisect_right(self.cum, idx) - 1
            start_in_series = idx - int(self.cum[si])
            s = self.series_tensors[si] # Get the tensor
            i = start_in_series
            x_vals = s[i : i + self.input_size]  # [input_size, 1]
            y = s[i + self.input_size : i + self.input_size + self.output_size].reshape(-1) # [output_size]
            if self.use_time_covariates and self.cov_tensor is not None:
                x_cov = self.cov_tensor[i : i + self.input_size]  # [input_size, C]
                X = torch.cat([x_vals, x_cov], dim=1)
            else:
                X = x_vals
                
            return X, y

class TimeseriesDataSet:
    def __init__(self, data: pd.DataFrame, granularity: Granularity, 
                 use_time_covariates, timestamp_column: str, 
                 train_val_ratio: float, scaler_class: Type[Scaler],
                 input_size: int, output_size: int, windows_in_memory: bool = True):
        self.data = data.copy()
        self.granularity = granularity
        self.timestamp_column = timestamp_column
        self.use_time_covariates = use_time_covariates
        self.train_val_ratio = train_val_ratio
        self.scaler_class = scaler_class
        self.input_size = input_size
        self.output_size = output_size
        self.windows_in_memory = windows_in_memory

        self._resample_dataframe()
        self._split_data()

        if self.use_time_covariates:
            self.train_time_feats = self._build_time_features(
                self.train_data[self.timestamp_column])
            self.val_time_feats = self._build_time_features(
                self.val_data[self.timestamp_column])
        else:
            self.train_time_feats = None
            self.val_time_feats = None

        self._scale_data()
        if self.windows_in_memory:
            self._create_scaled_datasets()

    def _resample_dataframe(self):
        self.data[self.timestamp_column] = pd.to_datetime(self.data[self.timestamp_column])
        value_columns = [col for col in self.data.columns if "value" in col]
        df_resampled = self.data.set_index(self.timestamp_column)
        if self.granularity == Granularity.HOURLY:
            df_resampled = df_resampled.resample('h')[value_columns].mean().reset_index()
        elif self.granularity == Granularity.DAILY:
            df_resampled = df_resampled.resample('d')[value_columns].mean().reset_index()
        elif self.granularity == Granularity.MONTHLY:
            df_resampled = df_resampled.resample('m')[value_columns].mean().reset_index()
        self.data = df_resampled

    def get_resampled_data(self) -> pd.DataFrame:
        return self.data

    def _split_data(self):
        n_total = len(self.data)
        self.n_train = int(n_total * self.train_val_ratio)
        self.train_data = self.data.iloc[:self.n_train]
        self.val_data = self.data.iloc[self.n_train:]

    def _build_time_features(self, timestamps: pd.Series) -> np.ndarray:
        ts = pd.to_datetime(timestamps)
        feats = []
        # month-of-year cyclic features
        m = ts.dt.month.values.astype(np.float32)
        feats += [np.sin(2*np.pi*m/12.0), np.cos(2*np.pi*m/12.0)]
        # day-of-week cyclic features for daily or hourly
        if self.granularity in (Granularity.HOURLY, Granularity.DAILY):
            dow = ts.dt.dayofweek.values.astype(np.float32)
            feats += [np.sin(2*np.pi*dow/7.0), np.cos(2*np.pi*dow/7.0)]
        # hour-of-day cyclic features for hourly
        if self.granularity == Granularity.HOURLY:
            hod = ts.dt.hour.values.astype(np.float32)
            feats += [np.sin(2*np.pi*hod/24.0), np.cos(2*np.pi*hod/24.0)]
        return np.vstack(feats).T.astype(np.float32)
    
    def get_train_time_features(self) -> np.ndarray:
        return self.train_time_feats
    def get_validation_time_features(self) -> np.ndarray:
        return self.val_time_feats
    
    def get_time_feature_count(self) -> int:
        count = 0
        if self.use_time_covariates:
            # month-of-year
            count += 2
            # day-of-week
            if self.granularity in (Granularity.HOURLY, Granularity.DAILY):
                count += 2
            # hour-of-day
            if self.granularity == Granularity.HOURLY:
                count += 2
        return count

    def _create_dataset(self, data, input_size, output_size, covariates=None) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = [], []

        for i in range(len(data) - input_size - output_size + 1):
            x_vals = data[i: i + input_size]
            if covariates is not None:
                x_cov = covariates[i: i + input_size]
                feature = np.concatenate([x_vals, x_cov], axis=1)
            else:
                feature = x_vals
            target = data[i + input_size: i + input_size + output_size]
            X.append(feature)
            y.append(target)

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32).reshape(-1, output_size)
        X_t = torch.tensor(X_arr, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)
        return X_t, y_t
    
    def _scale_data(self):
        self.split_scaled_dict = {}
        self.split_unscaled_dict = {}
        timeseries_dict = {
            col: self.data[col].values.reshape(-1, 1) 
            for col in self.data.columns if "value" in col
        }

        for key, series in timeseries_dict.items():
            train_unscaled = series[:self.n_train]
            validation_unscaled = series[self.n_train:]
            scaler = self.scaler_class()
            train_scaled = scaler.fit_transform(train_unscaled)
            validation_scaled = scaler.transform(validation_unscaled)

            if not np.isfinite(train_scaled).all() or not np.isfinite(validation_scaled).all():
                raise ValueError(f"{key} contains NaN/Inf after scaling.")

            self.split_scaled_dict[key] = {
                "train": train_scaled,
                "validation": validation_scaled,
                "scaler": scaler
            }
            self.split_unscaled_dict[key] = {
                "train": train_unscaled,
                "validation": validation_unscaled
            }

    def _create_scaled_datasets(self) -> dict:
        X_train, y_train, X_val, y_val = [], [], [], []
        for key in sorted(self.split_scaled_dict.keys()):
            train_scaled = self.split_scaled_dict[key]["train"]
            val_scaled = self.split_scaled_dict[key]["validation"]
            
            if self.use_time_covariates:
                X_train_series, y_train_series = self._create_dataset(train_scaled, self.input_size, self.output_size, self.train_time_feats)
                X_val_series, y_val_series = self._create_dataset(val_scaled, self.input_size, self.output_size, self.val_time_feats)
            else:
                X_train_series, y_train_series = self._create_dataset(train_scaled, self.input_size, self.output_size)
                X_val_series, y_val_series = self._create_dataset(val_scaled, self.input_size, self.output_size)

            X_train.append(X_train_series)
            y_train.append(y_train_series)
            X_val.append(X_val_series)
            y_val.append(y_val_series)

        self.X_train = torch.cat(X_train, dim=0)
        self.y_train = torch.cat(y_train, dim=0)
        self.X_val   = torch.cat(X_val, dim=0)
        self.y_val   = torch.cat(y_val, dim=0)

    def get_scaled_data(self, key: str) -> np.ndarray:
        return self.split_scaled_dict[key]
    
    def get_unscaled_data(self, key: str) -> np.ndarray:
        return self.split_unscaled_dict[key]
    
    def unscale_series(self, key: str, scaled_data: np.ndarray) -> np.ndarray:
        scaler = self.split_scaled_dict[key]["scaler"]
        return scaler.inverse_transform(scaled_data)

    def get_correlation_matrix(self) -> pd.DataFrame:
        value_columns = [col for col in self.data.columns if "value" in col]
        return self.data[value_columns].corr()
    
    def get_single_series_dataloader(
        self,
        key: str,
        split: str,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2
    ) -> torch.utils.data.DataLoader:
        series_scaled = self.split_scaled_dict[key][split]
        if self.use_time_covariates:
            if split == "train":
                covariates = self.train_time_feats
            else:
                covariates = self.val_time_feats
            X, y = self._create_dataset(series_scaled, self.input_size, self.output_size, covariates)
        else:
            X, y = self._create_dataset(series_scaled, self.input_size, self.output_size)
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def get_train_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2
    ) -> torch.utils.data.DataLoader:
        if self.windows_in_memory:
            dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        else:
            series_list = [self.split_scaled_dict[k]["train"] for k in sorted(self.split_scaled_dict.keys())]
            dataset = SlidingWindowDataset(
                series_list,
                self.train_time_feats,
                self.input_size,
                self.output_size,
                self.use_time_covariates,
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    
    def get_validation_dataloader(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2
    ) -> torch.utils.data.DataLoader:
        if self.windows_in_memory:
            dataset = torch.utils.data.TensorDataset(self.X_val, self.y_val)
        else:
            series_list = [self.split_scaled_dict[k]["validation"] for k in sorted(self.split_scaled_dict.keys())]
            dataset = SlidingWindowDataset(
                series_list,
                self.val_time_feats,
                self.input_size,
                self.output_size,
                self.use_time_covariates,
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )