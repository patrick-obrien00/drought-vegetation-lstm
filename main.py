import torch
import torch.nn as nn
import numpy as np
from model import PixelLSTM 
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


# === Load your saved data ===
X_test = np.load("data/X_test.npy")              # Shape: (T, P, F)
static_test = np.load("data/static_test.npy")    # Shape: (P, S)
y_test = np.load("data/y_test.npy")              # Shape: (T, P)

# === Preprocess ===
X_test = torch.tensor(X_test.transpose(1, 0, 2), dtype=torch.float32)  # (P, T, F)
static_test = torch.tensor(static_test, dtype=torch.float32)          # (P, S)

# === Define model architecture (must match training) ===
model = PixelLSTM(
    input_size=X_test.shape[2],
    static_size=static_test.shape[1],
    hidden_size=256,
    num_layers=3,
    dropout=0.3987832655685518
)

# === Load the weights ===
checkpoint = torch.load("data/model_checkpoint.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])  # ✅ just the model weights


model.eval()

# === Run inference ===
with torch.no_grad():
    predictions = model(X_test, static_test)  # Shape: (P, T)

# === Convert to NumPy if needed ===
predictions_np = predictions.numpy()

# === Evaluate (optional) ===
y_test = torch.tensor(y_test.transpose(1, 0), dtype=torch.float32)  # (P, T)
mse = torch.nn.functional.mse_loss(predictions, y_test)
print(f"Test MSE: {mse.item():.4f}")
