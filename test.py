import json
import numpy as np

# Datei laden
file_path = r"C:\Users\alexa\Documents\Webapp\data\production_history\production_summary_all.json"

# JSON-Datei öffnen und laden
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)
    
all_capacities = []
all_production_data = []

for wpp in data:
    all_capacities.append(wpp["GenerationUnitInstalledCapacity(MW)"])
    all_production_data.append(wpp["Production"])

capacity_values = []
production_values = []

for idx, production_data in enumerate(all_production_data):
    num_rows = len(production_data)
    capacity_values.append(np.full((num_rows, 1), all_capacities[idx]))
    production_values.append(np.array([entry[1] for entry in production_data]).reshape(-1, 1))

# Kombinieren aller Datensätze in einem großen Array
capacity_values = np.vstack(capacity_values)
production_values = np.vstack(production_values)

# Alle Werte auf zwei Nachkommastellen runden
capacity_values = np.round(capacity_values, decimals=2)
production_values = np.round(production_values, decimals=2)