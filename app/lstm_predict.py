import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))

def load_lstm():
    model = LSTMModel()
    model.load_state_dict(torch.load("lstm_model.pth", map_location="cpu"))
    model.eval()
    return model

def predict_lstm(model, seq):
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return model(seq).item()
