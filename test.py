import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import os

# Listen für alle Daten
all_turbine_types = []
all_hub_heights = []
all_capacities = []
all_commissioning_dates = []
all_production_data = []

# Daten der Jahre 2015 bis 2024 verarbeiten
for year in range(2015, 2025):
    file_path = f"data/WPPs+production+wind_{year}.json"

    # Prüfen, ob die Datei existiert
    if not os.path.isfile(file_path):
        print(f"Datei {file_path} nicht gefunden, überspringe...")
        continue

    # JSON-Datei laden
    with open(file_path, "r", encoding="utf-8") as file:
        WPP_production_wind = json.load(file)

    # Daten sammeln
    for wpp in WPP_production_wind:
        all_turbine_types.append(wpp["Turbine"])
        all_hub_heights.append(wpp["Hub_height"] if not pd.isna(wpp["Hub_height"]) else 100)
        all_capacities.append(wpp["Capacity"])
        all_commissioning_dates.append(wpp["Commission_date"] if wpp["Commission_date"] != "nan" else "2015/06")
        all_production_data.append(wpp["Production"])

# One-Hot-Encoding für Turbinentypen
encoder = OneHotEncoder(sparse_output=False)
turbine_types_onehot = encoder.fit_transform(np.array(all_turbine_types).reshape(-1, 1))

# Datumsformat korrigieren
all_commissioning_dates = [
    "2015/06" if date == "nan" else f"{date}/06" if isinstance(date, str) and "/" not in date else date
    for date in all_commissioning_dates
]

# In datetime konvertieren
standardised_dates = pd.to_datetime(all_commissioning_dates, format='%Y/%m')

# Berechnung des Alters
current_date = pd.Timestamp("2024-12-01")
ages = current_date.year * 12 + current_date.month - (standardised_dates.year * 12 + standardised_dates.month)

# Kombinierte Features und Outputs erstellen
combined_features = []
output = []

# Daten in Feature-Arrays konvertieren
for idx, production_data in enumerate(all_production_data):
    num_rows = len(production_data)

    # Wiederholungen für allgemeine Features
    turbine_type_repeated = np.tile(turbine_types_onehot[idx], (num_rows, 1))
    hub_height_repeated = np.full((num_rows, 1), all_hub_heights[idx])
    capacity_repeated = np.full((num_rows, 1), all_capacities[idx])
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

    # Füge die Daten hinzu
    combined_features.append(combined_chunk)
    output.append(production_values)

# Kombinieren aller Datensätze in einem großen Array
combined_features = np.vstack(combined_features)
output = np.vstack(output)

# For hyperparameter search: only retain a subset of the data
num_samples = 100000
random_indices = np.random.choice(combined_features.shape[0], num_samples, replace=False)
combined_features = combined_features[random_indices]
output = output[random_indices]

# Standardisierung der numerischen Features
scaler = StandardScaler()
numerical_columns = slice(turbine_types_onehot.shape[1], combined_features.shape[1] - 1)
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

# Ausgabe der Formen
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
        self.leaky_relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x
    
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import shutil
import os
from torch.utils.tensorboard import SummaryWriter

# Hyperparameter-Raum definieren
# param_space = {
#     "hidden_size": [32, 64, 128, 256],
#     "batch_size": [16, 32, 64],
#     "lr": [1e-2, 1e-3, 1e-4],
#     "number_epochs": [20, 50, 100],
# }
param_space = {
    "hidden_size": [16],
    "batch_size": [64],
    "lr": [1e-3],
    "number_epochs": [20],
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
params = random_search(param_space, n_trials)[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# KFold-Objekt
kf = KFold(n_splits=2, shuffle=True, random_state=1)
len_train_val_dataset = len(train_val_dataset)

# Ergebnis-Tracking
best_val_loss = float("inf")
best_params = None
results = []

input_size = train_features.shape[1]

# Pfad zum TensorBoard-Verzeichnis
log_dir = "runs"

# Löschen, wenn der Ordner existiert
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
if not os.path.exists(log_dir):
    print("Der Ordner wurde vollständig gelöscht.")
else:
    print("Der Ordner wurde nicht vollständig gelöscht.")


avg_val_loss = 0.0  # Durchschnittliche Validierungs-Fehler über die Folds

for fold, (train_idx, val_idx) in enumerate(kf.split(range(len_train_val_dataset)), 1):

    print(f"  Fold {fold}/{kf.n_splits}")
    writer = SummaryWriter(f"{log_dir}/fold_{fold}")

    # Modell
    example_input = torch.randn(params["batch_size"], input_size).to(device)
    model = MLP(input_size=input_size, hidden_size=params["hidden_size"], output_size=1).to(device)

    # Visualisierung der Modellarchitektur
    writer.add_graph(model, example_input)

    # Train- und Validierungsdaten erstellen
    train_fold_dataset = Subset(train_val_dataset, train_idx)
    val_fold_dataset = Subset(train_val_dataset, val_idx)

    train_loader = DataLoader(train_fold_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_fold_dataset, batch_size=params["batch_size"], shuffle=False)

    # Loss und Optimizer
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    # Training
    for epoch in range(params["number_epochs"]):
        print(f"    Epoch {epoch+1}/{params['number_epochs']}")
        model.train()
        training_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            training_loss += loss.item()

            optimizer.zero_grad()  # Gradienten zurücksetzen
            loss.backward()        # Gradienten berechnen
            optimizer.step()       # Parameter aktualisieren

        # Trainingsverlust protokollieren
        writer.add_scalar("Training Loss", training_loss / len(train_loader), epoch)

    # Validierung
    model.eval()
    fold_val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            val_outputs = model(batch_x)
            fold_val_loss += criterion(val_outputs, batch_y).item()

    fold_val_loss /= len(val_loader)
    writer.add_scalar("Validation Loss", fold_val_loss, 1)
    print(f"    Fold Validation Loss: {fold_val_loss:.4f}")
    avg_val_loss += fold_val_loss

    # TensorBoard schließen
    writer.close()

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