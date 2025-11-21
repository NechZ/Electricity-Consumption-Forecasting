import torch
from torch import nn
import numpy as np
import time
from typing import Dict, Callable
from torch.amp import autocast, GradScaler

from .models import BaseModel
from .preprocessing import TimeseriesDataSet

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class ModelTrainer:
    def __init__(self, model: BaseModel, device, optimizer, loss_fn, lr_scheduler,
                 train_loader, val_loader, early_stopper=None, warmup_scheduler=None, warmup_epochs=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = lr_scheduler
        self.early_stopper = early_stopper
        self.warmup_scheduler = warmup_scheduler
        self.warmup_epochs = warmup_epochs
        self.model.to(self.device)

    def _train_epoch(self) -> tuple:
        self.model.train()
        total_squared_error = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        total_absolute_error = torch.tensor(0.0, device=self.device, dtype=torch.float64)
        total_elements = 0
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            y_pred = self.model(X_batch)
            loss = self.loss_fn(y_pred, y_batch)
            errs = y_pred - y_batch
            total_squared_error += errs.pow(2).sum(dtype=torch.float64)
            total_absolute_error += errs.abs().sum(dtype=torch.float64)
            total_elements += y_batch.numel()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        rmse = torch.sqrt(total_squared_error / total_elements).item()
        mae = (total_absolute_error / total_elements).item()
        return rmse, mae

    def _validate_epoch(self) -> tuple:
        self.model.eval()
        with torch.no_grad():
            total_squared_error = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            total_absolute_error = torch.tensor(0.0, device=self.device, dtype=torch.float64)
            total_elements = 0
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                y_pred = self.model(X_batch)
                errs = y_pred - y_batch
                total_squared_error += errs.pow(2).sum(dtype=torch.float64)
                total_absolute_error += errs.abs().sum(dtype=torch.float64)
                total_elements += y_batch.numel()
            rmse = torch.sqrt(total_squared_error / total_elements).item()
            mae = (total_absolute_error / total_elements).item()
        return rmse, mae
    
    def fit(self, n_epochs: int, on_epoch_end: Callable[[int, dict], None] = None) -> dict:
        history = {
            "train_rmse": [], "train_mae": [],
            "val_rmse": [],   "val_mae": [],
            "learning_rate": [], "epoch_time": []
        }
        for epoch in range(1, n_epochs + 1):
            epoch_start = time.time()
            rmse, mae = self._train_epoch()
            val_rmse, val_mae = self._validate_epoch()
            epoch_end = time.time()
 
            if self.warmup_scheduler and self.warmup_epochs and epoch <= self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]['lr']
            history["train_rmse"].append(rmse)
            history["train_mae"].append(mae)
            history["val_rmse"].append(val_rmse)
            history["val_mae"].append(val_mae)
            history["learning_rate"].append(lr)
            history["epoch_time"].append(epoch_end - epoch_start)

            if on_epoch_end:
                on_epoch_end(epoch, {
                    "train_rmse": rmse,
                    "train_mae": mae,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "learning_rate": lr,
                    "epoch_time": epoch_end - epoch_start
                })

            if self.early_stopper and self.early_stopper.early_stop(val_rmse):
                break
        return history
    
    def get_model(self) -> BaseModel:
        return self.model
            
class TimeseriesForecaster:
    def __init__(self, model: BaseModel, device, input_size: int, output_size: int):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.model.to(self.device)
        self.model.eval()
    
    @torch.inference_mode()
    def predict(self, series_data, cov_data, batch_size) -> np.ndarray:
        """
        Generates predictions using non-overlapping output windows (tiling).
        Crucially, this now discards the final window if it is incomplete to 
        ensure output shape consistency.
        """
        inputs = []
        valid_split_points = range(self.input_size, len(series_data) + 1, self.output_size)
        
        for i in valid_split_points:
            if i - self.input_size < 0: 
                continue
            
            input_window_vals = series_data[i - self.input_size : i]
            
            if len(input_window_vals) != self.input_size:
                continue

            if cov_data is not None:
                input_window_covs = cov_data[i - self.input_size : i]
                feature = np.concatenate([input_window_vals, input_window_covs], axis=1)
            else:
                feature = input_window_vals
            inputs.append(feature)

        if not inputs:
            return np.array([])

        inputs_array = np.array(inputs)
        input_tensor = torch.tensor(inputs_array, dtype=torch.float32, device=self.device)
        
        preds = []
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i : i + batch_size]
            batch_preds = self.model(batch).cpu().numpy() # Shape: [batch, output_size]
            preds.append(batch_preds)
            
        preds = np.concatenate(preds, axis=0).flatten()
        return preds
    
    def predict_all_series(self, dataset: TimeseriesDataSet, batch_size: int) -> Dict[str, Dict[str, np.ndarray]]:
        all_preds = {}
        for key in sorted(dataset.split_scaled_dict):
            scaled = dataset.get_scaled_data(key)
            
            train_raw = self.predict(scaled["train"],
                                    dataset.get_train_time_features(), batch_size)
            
            val_raw   = self.predict(scaled["validation"],
                                    dataset.get_validation_time_features(), batch_size)

            train = dataset.unscale_series(key, train_raw.reshape(-1, 1)).flatten()
            val   = dataset.unscale_series(key, val_raw.reshape(-1, 1)).flatten()
            
            all_preds[key] = {"train": train, "validation": val}
        return all_preds