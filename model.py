import torch
import torch.nn as nn

class PixelLSTM(nn.Module):
    def __init__(self, input_size, static_size, hidden_size, num_layers, dropout=0.0):
        super(PixelLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        self.static_to_hidden = nn.Linear(static_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, x_static, return_hidden=False):
        batch_size = x.size(0)
        h0_proj = self.static_to_hidden(x_static)
        h0 = h0_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = h0.clone()

        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.output_layer(lstm_out).squeeze(-1)

        if return_hidden:
            return out, lstm_out
        else:
            return out
