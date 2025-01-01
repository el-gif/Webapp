import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
import json

# Lade die synthetischen Daten
with open("data/synthetic_data.json", "r", encoding="utf-8") as f:
    synthetic_data = json.load(f)

# Extrahiere Windgeschwindigkeiten, Kapazitäten und Leistungen
wind_speeds = []
capacities = []
powers = []

for entry in synthetic_data:
    capacity = entry["Capacity"]
    for production_entry in entry["Production"]:
        wind_speed = production_entry[2]
        power = production_entry[1]

        wind_speeds.append(wind_speed)
        capacities.append(capacity)
        powers.append(power)

# Konvertiere zu numpy Arrays
wind_speeds = np.array(wind_speeds)
capacities = np.array(capacities)
powers = np.array(powers)

# Definiere die Zielgitter für die Interpolation
unique_wind_speeds = np.linspace(0, 30, 250)
unique_capacities = np.linspace(0, 15, 100)
wind_speeds_grid, capacities_grid = np.meshgrid(unique_wind_speeds, unique_capacities)

# Konvertiere Kapazitäten so, dass sie dieselbe Länge wie die Windgeschwindigkeiten haben
expanded_capacities = np.tile(capacities, len(synthetic_data[0]["Production"][0][0]))

# Prüfe, ob die Dimensionen jetzt übereinstimmen
assert len(expanded_capacities) == len(wind_speeds.flatten()), "Die Dimensionen von wind_speeds und expanded_capacities stimmen nicht überein!"

# Interpolation vorbereiten
points = np.vstack((wind_speeds.flatten(), expanded_capacities)).T
values = powers.flatten()

# Interpoliere die Werte der Power Curves auf das Zielgitter
powers_grid = griddata(points, values, (wind_speeds_grid, capacities_grid), method='linear', fill_value=0)

# Flache Ebene: capacity = power
z_flat = capacities_grid  # Ebene durch Kapazitäten dargestellt

# Rest des Codes bleibt gleich
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])

color_mask = np.where(powers_grid > z_flat, 1, 0)
cmap = plt.get_cmap("coolwarm")
norm = plt.Normalize(vmin=0, vmax=1)
facecolors = cmap(norm(color_mask))

surf = ax.plot_surface(wind_speeds_grid, capacities_grid, powers_grid, facecolors=facecolors, edgecolor='none', alpha=0.9)
ax.plot_surface(wind_speeds_grid, capacities_grid, z_flat, color='gray', alpha=0.3, edgecolor='none')

ax.set_xlim(ax.get_xlim()[::-1])
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Capacity (MW)")
ax.set_zlabel("Power (MW)")
ax.set_title("3D Plot of Power Curves; Derived from Synthetic Data")

legend_elements = [
    Line2D([0], [0], color='red', lw=4, label='Power > Capacity'),
    Line2D([0], [0], color='blue', lw=4, label='Power <= Capacity')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.show()
