import numpy as np
import random
import string
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
from datetime import datetime

###### data #######

print("load data")

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
hub_heights = np.random.normal(100, 10, 100) # mean, standard deviation, number of samples
commission_dates = np.random.uniform(1990, 2024, 100)
rotor_diameters = np.random.normal(50, 5, 100)
capacities = np.random.normal(2000, 300, 100)
wind_power = np.random.normal(1000, 150, 100)

ages = datetime.now().year*12 + datetime.now().month - commission_dates*12
points = np.column_stack((lons, lats))  # Eingabekoordinaten als (lon, lat)-Paare

wind_speeds_plants = []

# interpolation functions (in contrast to interp2d, griddata also works with irregularly arranged grid data points)
for t in range(len(times)):
    print(f"time step {t}")
    wind_speeds_plants.append(griddata(points, wind_speeds[t,:,:].flatten(), (lons_plants, lats_plants), method='nearest'))

print("loop finished")
wind_speeds_plants = np.array(wind_speeds_plants)  # Convert to numpy array (time x plants)

print("data loaded")

###### model ########

print("prepare model")

# Daten vorbereiten
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# One-Hot-Encoding für turbine_types
encoder = OneHotEncoder(sparse_output=False)
turbine_types_onehot = encoder.fit_transform(turbine_types.reshape(-1, 1))

# Skalierung der Features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(np.stack([hub_heights, ages, rotor_diameters, capacities], axis=1))
wind_speeds_plants_scaled = scaler.fit_transform(wind_speeds_plants.reshape(-1, wind_speeds_plants.shape[-1])).reshape(wind_speeds_plants.shape)

class WindPowerDataset(Dataset):
    def __init__(self, wind_speeds, wind_power, static_features, turbine_types_onehot, sequence_length):
        self.wind_speeds = wind_speeds
        self.wind_power = wind_power
        self.static_features = static_features
        self.turbine_types_onehot = turbine_types_onehot
        self.sequence_length = sequence_length

    def __len__(self):
        return self.wind_speeds.shape[0] - self.sequence_length + 1

    def __getitem__(self, index):
        # Dynamische Sequenz erstellen
        wind_speed_seq = self.wind_speeds[index:index + self.sequence_length].reshape(
            self.sequence_length, -1, 1
        )  # Form: (sequence_length, n_plants, 1)
        
        static_features_seq = np.tile(
            np.concatenate([self.turbine_types_onehot, self.static_features], axis=1), 
            (self.sequence_length, 1, 1)
        ).reshape(self.sequence_length, -1, self.static_features.shape[1])  # Form: (sequence_length, n_plants, n_features_combined)
        
        # Eingabe-Features kombinieren
        x = np.concatenate([static_features_seq, wind_speed_seq], axis=-1)  # Form: (sequence_length, n_plants, total_features)
        y = self.wind_power[index + self.sequence_length - 1]  # Zielwert ist der letzte Punkt der Sequenz

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Dataset und DataLoader erstellen
sequence_length = 24  # Sequenzlänge
dataset = WindPowerDataset(
    wind_speeds=wind_speeds_plants_scaled,
    wind_power=wind_power,
    static_features=features_scaled,
    turbine_types_onehot=turbine_types_onehot,
    sequence_length=sequence_length
) # Ruft dataset.__init__() auf

# LSTM-Modell definieren
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.leaky_relu(self.fc(out[:, -1, :]))
        return out

number_epochs = 50

# Cross-Validation
kf = KFold(n_splits=10)
fold = 1

print("model prepared")

for train_idx, val_idx in kf.split(dataset):
    print(f"Fold {fold}/{kf.n_splits}")
    fold += 1

    # Train- und Validierungsdaten erstellen
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True) # ruft dataset.__getitem__() auf
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False) # ruft dataset.__getitem__() auf

    # Modell, Loss und Optimizer
    model = LSTM(input_size=dataset[0][0].shape[1], hidden_size=64, num_layers=2, output_size=1).to(device) # Ruft model.__init__() auf
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(number_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x) # Ruft model.forward() auf
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
torch.save(model.state_dict(), "lstm_wind_power_model.pth")