import numpy as np
import random
import string
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Daten vorbereiten
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Daten laden
lons = np.load("data/weather_history/COSMO_REA6/lons.npy").flatten()
lats = np.load("data/weather_history/COSMO_REA6/lats.npy").flatten()
times = np.load("data/weather_history/COSMO_REA6/times.npy")
pressures = np.load("data/weather_history/COSMO_REA6/pressure_values.npy")
temperatures = np.load("data/weather_history/COSMO_REA6/temperature_values.npy")
wind_speeds = np.load("data/weather_history/COSMO_REA6/wind_speed_values.npy")

# 20 turbine types
turbine_type_database = [''.join(random.choices(string.ascii_letters + string.digits, k=5)) for _ in range(20)]

# coordinates for Europe (more restrictive, because the covered area in the COSMO_REA6 dataset is not a rectangle)
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

pressures_plants = []
temperatures_plants = []
wind_speeds_plants = []

points = np.column_stack((lons, lats))  # Eingabekoordinaten als (lon, lat)-Paare

# interpolation functions (in contrast to interp2d, griddata also works with irregularly arranged grid data points)
for t in range(len(times)):
    pressures_plants.append(griddata(points, pressures.flatten(), (lons_plants, lats_plants), method='cubic'))
    temperatures_plants.append(griddata(points, temperatures.flatten(), (lons_plants, lats_plants), method='cubic'))
    wind_speeds_plants.append(griddata(points, wind_speeds.flatten(), (lons_plants, lats_plants), method='cubic'))

# One-Hot-Encoding für turbine_types
encoder = OneHotEncoder(sparse=False)
turbine_types_onehot = encoder.fit_transform(turbine_types.reshape(-1, 1))

# Eingabedaten vorbereiten
features = np.stack([hub_heights, commission_dates, rotor_diameters, capacities], axis=1)
weather_features = np.stack([pressures_plants[:, :, 0], temperatures_plants[:, :, 0], wind_speeds_plants[:, :, 0]], axis=-1)
inputs = np.concatenate([turbine_types_onehot, features, weather_features], axis=1)

# Tensor konvertieren
inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
targets = torch.rand(100, 1)  # Dummy-Zielwerte (später durch echte Werte ersetzen)
targets = torch.tensor(targets, dtype=torch.float32).to(device)

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

# Hyperparameter
input_size = inputs.shape[1]
hidden_size = 4
num_layers = 3
output_size = 1
learning_rate = 0.001
batch_size = 10
num_epochs = 100

# Modell, Loss und Optimizer
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(inputs.size(0))

    for i in range(0, inputs.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = inputs[indices], targets[indices]

        outputs = model(batch_x.unsqueeze(1))
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Modell speichern
torch.save(model.state_dict(), "lstm_wind_power_model.pth")