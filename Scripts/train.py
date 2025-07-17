
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
import wfdb
import ast
import wandb

# Initialize Weights & Biases
wandb.init(project="ptbxl-ecg-transformer", config={
    "epochs": 5,
    "batch_size": 32,
    "lr": 0.001,
    "model": "ECG_Transformer",
    "optimizer": "Adam"
})

# Model Definition
class ECG_Transformer(nn.Module):
    def __init__(self, seq_len=5000, num_features=12, d_model=32, nhead=2, num_layers=2, num_classes=10):
        super(ECG_Transformer, self).__init__()
        self.input_linear = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# Dataset
class PTBXL_Dataset(Dataset):
    def __init__(self, records, labels, signal_path, sr=500):
        self.records = records
        self.labels = labels
        self.signal_path = signal_path
        self.sr = sr

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        label = self.labels[idx]
        signal, _ = wfdb.rdsamp(os.path.join(self.signal_path, rec))
        signal = signal[:5000] if self.sr == 500 else signal[:1000]
        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return signal, label

# Load Metadata
df = pd.read_csv('data/raw/ptbxl/ptbxl_database.csv')
scp_statements = pd.read_csv('data/raw/ptbxl/scp_statements.csv')

df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
df['diagnostic_superclass'] = df['scp_codes'].apply(lambda x: list(x.keys()))
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['diagnostic_superclass'])

records = df['filename_hr'].str.replace('.hea', '', regex=False).values
train_rec, test_rec, y_train, y_test = train_test_split(records, y, test_size=0.2, random_state=42)

train_ds = PTBXL_Dataset(train_rec, y_train, 'data/raw/ptbxl/records500')
test_ds = PTBXL_Dataset(test_rec, y_test, 'data/raw/ptbxl/records500')
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECG_Transformer(num_classes=len(mlb.classes_)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)

wandb.watch(model)

for epoch in range(wandb.config.epochs):
    model.train()
    total_loss = 0
    for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        wandb.log({"train_loss": loss.item()})
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluate
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for signals, labels in test_loader:
        signals, labels = signals.to(device), labels.to(device)
        outputs = model(signals)
        y_true.append(labels.cpu().numpy())
        y_pred.append(torch.sigmoid(outputs).cpu().numpy())

macro_auc = roc_auc_score(np.vstack(y_true), np.vstack(y_pred), average='macro')
print("Classification Report:")
print(classification_report(np.vstack(y_true) > 0.5, np.vstack(y_pred) > 0.5, target_names=mlb.classes_))
wandb.log({"macro_auc": macro_auc})

# Save Model
torch.save({'model_state_dict': model.state_dict(), 'classes': mlb.classes_}, 'ecg_transformer.pt')
wandb.save('ecg_transformer.pt')
