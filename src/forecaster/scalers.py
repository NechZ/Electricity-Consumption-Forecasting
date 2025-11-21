from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler

class Scaler(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def inverse_transform(self, X_scaled):
        raise NotImplementedError
    
class Pipeline():
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        for name, step in self.steps:
            X = step.fit_transform(X, y)
        return self

    def transform(self, X, y=None):
        for name, step in self.steps:
            X = step.transform(X, y)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform(self, X_scaled):
        for name, step in reversed(self.steps):
            X_scaled = step.inverse_transform(X_scaled)
        return X_scaled

class CenteredScaler(Scaler):
    def __init__(self):
        self.mean_ = None

    def fit(self, X, y=None):
        # compute mean per-feature (works for 1D and 2D arrays)
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X, y=None):
        return X - self.mean_
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        return X_scaled + self.mean_

class LogScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(X + 1e-8)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        return np.exp(X_scaled) - 1e-8

class SqrtScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.sqrt(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        return X_scaled ** 2
    
class NoneScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        return X_scaled

class ArcsinhScaler(Scaler):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.arcsinh(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        return np.sinh(X_scaled)

class LogStandardScaler(Scaler):
    def __init__(self):
        self.pipeline = Pipeline([
            ('log_scaler', LogScaler()),
            ('standard_scaler', StandardScaler())
        ])
    
    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)
    
    def inverse_transform(self, X_scaled):
        return self.pipeline.inverse_transform(X_scaled)

class CenteredLogScaler(Scaler):
    def __init__(self):
        self.pipeline = Pipeline([
            ('log_scaler', LogScaler()),
            ('centered_scaler', CenteredScaler())
        ])
    
    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)
    
    def inverse_transform(self, X_scaled):
        return self.pipeline.inverse_transform(X_scaled)
    
class CenteredSqrtScaler(Scaler):
    def __init__(self):
        self.pipeline = Pipeline([
            ('sqrt_scaler', SqrtScaler()),
            ('centered_scaler', CenteredScaler())
        ])
    
    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)
    
    def inverse_transform(self, X_scaled):
        return self.pipeline.inverse_transform(X_scaled)
    
class CenteredLogArcSinhScaler(Scaler):
    def __init__(self):
        self.pipeline = Pipeline([
            ('log_scaler', LogScaler()),
            ('centered_scaler', CenteredScaler()),
            ('sqrt_scaler', ArcsinhScaler()),
        ])
    
    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)
    
    def inverse_transform(self, X_scaled):
        return self.pipeline.inverse_transform(X_scaled)
    
class StandardArcSinhScaler(Scaler):
    def __init__(self):
        self.pipeline = Pipeline([
            ('arcsinh_scaler', ArcsinhScaler()),
            ('standard_scaler', StandardScaler())
        ])
    
    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)
    
    def inverse_transform(self, X_scaled):
        return self.pipeline.inverse_transform(X_scaled)