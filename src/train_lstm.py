import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


X = np.load("X_seq.npy")
y = np.load("y_seq.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=128,
    shuffle=True
)


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return torch.sigmoid(out)

model = LSTMModel()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(5):

    total_loss = 0

    for xb, yb in train_loader:

        preds = model(xb).squeeze()
        loss = loss_fn(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss {total_loss:.4f}")

print("Training done!")

from sklearn.metrics import accuracy_score, roc_auc_score

model.eval()

with torch.no_grad():
    preds = model(X_test).squeeze().numpy()

pred_labels = (preds > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test.numpy(), pred_labels))
print("ROC AUC:", roc_auc_score(y_test.numpy(), preds))

torch.save(model.state_dict(), "lstm_model.pth")