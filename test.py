import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder

# Listen für alle Daten
all_turbine_types = []
all_hub_heights = []
all_capacities = []
all_commissioning_dates = []
all_production_data = []

# JSON-Datei laden
with open(f"data/WPPs+production+wind.json", "r", encoding="utf-8") as file:
    WPP_production_wind = json.load(file)

# Daten sammeln
for wpp in WPP_production_wind:
    all_turbine_types.append(wpp["Turbine"])
    all_hub_heights.append(wpp["Hub_height"] if not pd.isna(wpp["Hub_height"]) else 100)
    all_capacities.append(wpp["Capacity"])
    all_commissioning_dates.append(wpp["Commission_date"] if wpp["Commission_date"] != "nan" else "2015/06")
    all_production_data.append(wpp["Production"])

# NaN-Werte in Turbinentypen durch eindeutige Namen ersetzen
nan_counter = 1
for idx, turbine in enumerate(all_turbine_types):
    if pd.isna(turbine):
        all_turbine_types[idx] = f"nan{nan_counter}"
        nan_counter += 1

turbine_types_set = set(all_turbine_types)
print(turbine_types_set)

from windpowerlib import WindTurbine
from windpowerlib import data as wt
import pandas as pd

# Initialisierung der Liste für gefundene Power Curves
power_curves = {}

# Turbinendaten abrufen
turbine_data = wt.get_turbine_types(print_out=False)

# Schleife über alle Turbinentypen
for turbine_type in turbine_types_set:
    try:

        # Turbine filtern
        matching_turbine = turbine_data[
            turbine_data["turbine_type"].str.lower() == turbine_type.lower()
        ]

        if not matching_turbine.empty:

            # Power Curve abrufen
            power_curve = matching_turbine.power_curve["value"]
            wind_speeds = matching_turbine.power_curve["wind_speed"]

            # Speichern der Daten
            power_curves[turbine_type] = pd.DataFrame({
                "Wind Speed (m/s)": wind_speeds,
                "Power (kW)": power_curve,
            })
            print(f"Power Curve für {turbine_type} gefunden.")
        else:
            print(f"Turbine {turbine_type} nicht in der Datenbank gefunden.")
    except Exception as e:
        print(f"Fehler bei {turbine_type}: {e}")

# Ausgabe als DataFrame oder Excel
for turbine, curve in power_curves.items():
    print(f"{turbine} Power Curve:\n", curve)
