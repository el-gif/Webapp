import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

# Laden der JSON-Datei
with open("data/WPPs+production+wind 2015.json", "r", encoding="utf-8") as file:
    WPP_production_wind = json.load(file)

# Extrahieren von allgemeinen Features für jedes Windkraftwerk
turbine_types = [wpp["Turbine"] for wpp in WPP_production_wind]
hub_heights = [wpp["Hub_height"] for wpp in WPP_production_wind]
capacities = [wpp["Capacity"] for wpp in WPP_production_wind]
commissioning_dates = [wpp["Commission_date"] for wpp in WPP_production_wind]

# One-Hot-Encoding für Turbinen-Typen
encoder = OneHotEncoder(sparse_output=False)
turbine_types_onehot = encoder.fit_transform(np.array(turbine_types).reshape(-1, 1))

# Berechnung der Alter der Windkraftwerke
standardised_dates = pd.to_datetime(commissioning_dates, format='%Y/%m')
current_date = pd.Timestamp("2024-12-01")
ages = current_date.year * 12 + current_date.month - (standardised_dates.year * 12 + standardised_dates.month)

# Listen für die kombinierten Features und die Outputs
combined_features = []
output = []

# Verarbeitung der JSON-Daten
for idx, wpp in enumerate(WPP_production_wind):
    production_data = wpp["Production"]  # Liste von [production_value, wind_speed]

    num_rows = len(production_data)  # Anzahl der Produktionsdatensätze

    # Erstelle Wiederholungen für allgemeine Features
    turbine_type_repeated = np.tile(turbine_types_onehot[idx], (num_rows, 1))
    hub_height_repeated = np.full((num_rows, 1), hub_heights[idx])
    capacity_repeated = np.full((num_rows, 1), capacities[idx])
    age_repeated = np.full((num_rows, 1), ages[idx])

    # Extrahiere Produktionswerte und Windgeschwindigkeiten
    production_values = np.array([entry[1] for entry in production_data]).reshape(-1, 1)
    wind_speeds = np.array([entry[2] for entry in production_data]).reshape(-1, 1)

    # Kombiniere alle Features
    combined_chunk = np.hstack((
        turbine_type_repeated,  # One-Hot Turbinen-Typ
        hub_height_repeated,    # Nabenhöhe
        capacity_repeated,      # Kapazität
        age_repeated,           # Alter
        wind_speeds             # Windgeschwindigkeit
    ))

    # Füge die Daten zum Gesamtdatensatz hinzu
    combined_features.append(combined_chunk)
    output.append(production_values)

# combine all chunks into one array
combined_features = np.vstack(combined_features)
output = np.vstack(output)

# Konvertieren in NumPy-Arrays
combined_features = np.array(combined_features)
output = np.array(output)

# Standardisierung der numerischen Features
scaler = StandardScaler()
numerical_columns = slice(turbine_types_onehot.shape[1], combined_features.shape[1] - 1)  # Numerische Spalten
combined_features[:, numerical_columns] = scaler.fit_transform(combined_features[:, numerical_columns])

# Trainings- und Testaufteilung
train_features, test_features, train_targets, test_targets = train_test_split(
    combined_features, output, test_size=0.25, random_state=1
)

# Dataset-Klasse für PyTorch
class WindPowerDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Erstellung der PyTorch-Datasets
train_val_dataset = WindPowerDataset(train_features, train_targets)
test_dataset = WindPowerDataset(test_features, test_targets)

# Ausgabe
print("Train Features Shape:", train_features.shape)
print("Train Targets Shape:", train_targets.shape)
print("Test Features Shape:", test_features.shape)
print("Test Targets Shape:", test_targets.shape)

import torch.nn as nn

# MLP-Modell definieren
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# Hyperparameter-Raum definieren
param_space = {
    "hidden_size": [32, 64, 128, 256],
    "batch_size": [16, 32, 64],
    "lr": [1e-2, 1e-3, 1e-4],
    "number_epochs": [20, 50, 100],
}

# Funktion zur Auswahl eines zufälligen Parametersets
def random_search(param_space, n_trials):
    trials = []
    for _ in range(n_trials):
        trial = {key: random.choice(values) for key, values in param_space.items()}
        trials.append(trial)
    return trials

# Generiere zufällige Parameterkombinationen
n_trials = 1
random_params = random_search(param_space, n_trials)

# Gerät auswählen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# KFold-Objekt
kf = KFold(n_splits=2, shuffle=True, random_state=1)
len_train_val_dataset = len(train_val_dataset)

# Ergebnis-Tracking
best_val_loss = float("inf")
best_params = None
results = []

input_size = train_features.shape[1]

# Random Search
for trial_idx, params in enumerate(random_params):
    print(f"Trial {trial_idx+1}/{n_trials} - Parameters: {params}")
    avg_val_loss = 0.0  # Durchschnittliche Validierungs-Fehler über die Folds

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len_train_val_dataset)), 1):
        print(f"  Fold {fold}/{kf.n_splits}")

        # Train- und Validierungsdaten erstellen
        train_fold_dataset = Subset(train_val_dataset, train_idx)
        val_fold_dataset = Subset(train_val_dataset, val_idx)

        train_loader = DataLoader(train_fold_dataset, batch_size=params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_fold_dataset, batch_size=params["batch_size"], shuffle=False)

        # Modell, Loss und Optimizer
        model = MLP(input_size=input_size, hidden_size=params["hidden_size"], output_size=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])

        # Training
        for epoch in range(2): #range(params["number_epochs"]):
            print(f"    Epoch {epoch+1}/{params['number_epochs']}")
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Validierung
        model.eval()
        fold_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_outputs = model(batch_x)
                fold_val_loss += criterion(val_outputs, batch_y).item()

        fold_val_loss /= len(val_loader)
        print(f"    Fold Validation Loss: {fold_val_loss:.4f}")
        avg_val_loss += fold_val_loss

    avg_val_loss /= kf.n_splits
    print(f"  Trial Average Validation Loss: {avg_val_loss:.4f}")

    # Ergebnisse speichern
    results.append({"params": params, "avg_val_loss": avg_val_loss})

    # Bestes Ergebnis aktualisieren
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_params = params

print(f"\nBest Parameters: {best_params}")
print(f"Best Validation Loss: {best_val_loss:.4f}")

# Test mit besten Parametern
model = MLP(input_size=input_size, hidden_size=best_params["hidden_size"], output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])

train_loader = DataLoader(train_val_dataset, batch_size=best_params["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)

# Training mit besten Parametern
for epoch in range(best_params["number_epochs"]):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test mit Testdaten
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        test_outputs = model(batch_x)
        test_loss += criterion(test_outputs, batch_y).item()

test_loss /= len(test_loader)
print(f"\nTest Loss: {test_loss:.4f}")

# Modell speichern
torch.save(model.state_dict(), "mlp_wind_power_model_best.pth")