import numpy as np
import random
import string
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from datetime import datetime

# Daten laden
lons = np.load("data/weather_history/COSMO_REA6/lons.npy").flatten()
lats = np.load("data/weather_history/COSMO_REA6/lats.npy").flatten()
times = np.load("data/weather_history/COSMO_REA6/times.npy")
wind_speeds = np.load("data/weather_history/COSMO_REA6/wind_speeds.npy")

# 20 turbine types
turbine_type_database = [''.join(random.choices(string.ascii_letters + string.digits, k=5)) for _ in range(20)]

# coordinates for Europe (more restrictive, because the covered area in the COSMO_REA6 dataset is not rectangular)
lat_min, lat_max = 35, 72
lon_min, lon_max = -10, 35

# 100 wind power plants
lons_plants = np.random.uniform(lon_min, lon_max, 100)
lats_plants = np.random.uniform(lat_min, lat_max, 100)
turbine_types = np.random.choice(turbine_type_database, size=100)
hub_heights = np.random.normal(100, 10, 100)  # mean, standard deviation, number of samples
commission_dates = np.random.uniform(1990, 2024, 100)
rotor_diameters = np.random.normal(50, 5, 100)
capacities = np.random.normal(2000, 300, 100)
wind_power = np.random.normal(1000, 150, (48, 100))

ages = datetime.now().year * 12 + datetime.now().month - commission_dates * 12
points = np.column_stack((lons, lats))  # Eingabekoordinaten als (lon, lat)-Paare

wind_speeds_plants = []

# interpolation functions
for t in range(len(times)):
    wind_speeds_plants.append(
        griddata(points, wind_speeds[t, :, :].flatten(), (lons_plants, lats_plants), method="nearest")
    )

wind_speeds_plants = np.array(wind_speeds_plants)  # Convert to numpy array (time x plants)

# One-Hot-Encoding für Turbinentypen
encoder = OneHotEncoder(sparse_output=False)
turbine_types_onehot = encoder.fit_transform(turbine_types.reshape(-1, 1))

# Skalierung der Features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(np.stack([hub_heights, ages, rotor_diameters, capacities], axis=1))
wind_speeds_scaled = scaler.fit_transform(wind_speeds_plants)

# Flatten der Windgeschwindigkeit
wind_speeds_plants_scaled_flat = wind_speeds_scaled.reshape(-1, 1)  # Shape: (48 * 100, 1)

# Wiederhole statische Features und Turbinentypen
turbine_types_repeated = np.repeat(turbine_types_onehot, repeats=48, axis=0)  # Shape: (48 * 100, ...)
features_repeated = np.repeat(features_scaled, repeats=48, axis=0)  # Shape: (48 * 100, ...)

# Kombinierte Eingabedaten
combined_features = np.concatenate([turbine_types_repeated, features_repeated, wind_speeds_plants_scaled_flat], axis=1)

# Zielwerte flatten
wind_power_flat = wind_power.T.flatten()  # Shape: (48 * 100,)

# Windkraftwerke aufteilen (Train/Test)
plant_indices = np.arange(100)
train_plants, test_plants = train_test_split(plant_indices, test_size=0.25, random_state=42)

# Trainingsdaten und Testdaten
train_indices = np.isin(np.arange(100), train_plants)
test_indices = np.isin(np.arange(100), test_plants)

train_features = combined_features[train_indices.repeat(48)]
train_targets = wind_power_flat[train_indices.repeat(48)]

test_features = combined_features[test_indices.repeat(48)]
test_targets = wind_power_flat[test_indices.repeat(48)]

# Dataset erstellen
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

train_dataset = WindPowerDataset(features=train_features, targets=train_targets)
test_dataset = WindPowerDataset(features=test_features, targets=test_targets)

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

# Trainingseinstellungen
number_epochs = 50
batch_size = 10
kf = KFold(n_splits=10)

# Cross-Validation
print("Start training")
fold = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for train_idx, val_idx in kf.split(range(len(train_dataset))):  # Indizes für KFold
    print(f"Fold {fold}/{kf.n_splits}")
    fold += 1

    # Train- und Validierungsdaten erstellen
    train_fold_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    val_fold_dataset = torch.utils.data.Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False)

    # Modell, Loss und Optimizer
    input_size = train_dataset[0][0].shape[0]
    model = MLP(input_size=input_size, hidden_size=64, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(number_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validierung
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_outputs = model(batch_x)
                val_loss += criterion(val_outputs.squeeze(), batch_y).item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

# Modell speichern
torch.save(model.state_dict(), "mlp_wind_power_model.pth")

# Test mit Testdaten
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        test_outputs = model(batch_x)
        test_loss += criterion(test_outputs.squeeze(), batch_y).item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")
