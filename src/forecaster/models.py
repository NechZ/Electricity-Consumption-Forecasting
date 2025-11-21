import torch
from torch import nn

class BaseModel(nn.Module):
    """
    Abstract base class for all models.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *inputs):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def count_parameters(self):
        """
        Counts the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class LSTM(BaseModel):
    """
    An LSTM-based model for time series forecasting.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the output of the last time step
        y = self.fc(out)
        return y
    
class LSTMAttention(BaseModel):
    """
    An LSTM-based model with attention mechanism for time series forecasting.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_attention=False):
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        e = self.score(out).squeeze(-1)
        alpha = self.softmax(e)
        context = torch.sum(out * alpha.unsqueeze(-1), dim=1)
        y = self.fc(context)
        if return_attention:
            return y, e
        return y

class GRU(BaseModel):
    """
    A GRU-based model for time series forecasting.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Take the output of the last time step
        y = self.fc(out)
        return y
    
class GRUAttention(BaseModel):
    """
    A GRU-based model with attention mechanism for time series forecasting.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(GRUAttention, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_attention=False):
        out, _ = self.gru1(x)
        out = self.layer_norm1(out)
        e = self.score(out).squeeze(-1)
        alpha = self.softmax(e)
        context = torch.sum(out * alpha.unsqueeze(-1), dim=1)
        y = self.fc(context)
        if return_attention:
            return y, e
        return y