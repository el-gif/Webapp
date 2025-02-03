import pandas as pd
import json

# Laden der Daten
df_wind_power = pd.read_excel("data/WPPs/Windfarms_Europe_20241123.xlsx", sheet_name="Windfarms")
df_assignment = pd.read_excel("data/Assignment_manual.xlsx", sheet_name="Sheet1")
with open(r"C:\Users\alexa\Documents\Webapp\data\production_history\production_summary_all.json", "r") as file:
    df_json = json.load(file)
    
# Filtere nur Zeilen, bei denen "ID_The-Wind-Power" nicht "not found" ist
df_assignment = df_assignment[df_assignment["ID_The-Wind-Power"] != "not found"]

# set with unique generation unit codes
generation_unit_code_set = set(df_assignment['GenerationUnitCode'])

# Extrahiere und entpacke alle gültigen IDs aus der Spalte "ID_The-Wind-Power"
def extract_ids(value):
    # Überprüfen, ob der Wert eine Liste ist, und ggf. in einzelne IDs zerlegen
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        return eval(value)  # Konvertiert die Zeichenkette in eine Liste
    elif isinstance(value, (int, str)):
        return [int(value)]  # Einzelne IDs werden in eine Liste gewandelt
    return []

valid_ids = set()
df_assignment["ID_The-Wind-Power"].apply(lambda x: valid_ids.update(extract_ids(x)))

df_filtered = df_wind_power[df_wind_power['ID'].isin(valid_ids)].copy()

print("number WPPs:", len(valid_ids))

production_data = [] # neues JSON-File mit Produktionsdaten für die WPPs
temporal_wpps = [] # WPPs, die temporär gespeichert werden, um sie später zu aktualisieren

# Gehe durch jede Zeile der Assignment-Datei und füge Produktionsdaten hinzu
for _, row in df_assignment.iterrows():

    production_array = df_json[row['JSON-ID']]['Production']
    capacity = row['GenerationUnitInstalledCapacity(MW)']

    capacities = []
    ids_in_row = extract_ids(row["ID_The-Wind-Power"])
    for index, id in enumerate(ids_in_row):
        current_index = df_filtered.loc[df_filtered['ID'] == id].index[0]
        capacities.append(df_filtered.at[current_index, "Total power"])
        capacity_total = sum(capacities)

    for index, id in enumerate(ids_in_row):
        current_index = df_filtered.loc[df_filtered['ID'] == id].index[0]

        # scale capacity and production in proportion to their occurences in the The Wind Power database (can be different turbine types)
        capacity_instance = capacity * capacities[index] / capacity_total
        production_array_instance = [
            [entry[0], entry[1] * (capacities[index] / capacity_total)]
            for entry in df_json[row['JSON-ID']]['Production']
        ]


        # Daten für das Windkraftwerk hinzufügen
        row_data = {
            'Name': row['GenerationUnitName'], # from assignment file
            'ID_The-Wind-Power': id, # from assignment file
            'JSON-ID': row['JSON-ID'], # from assignment file
            'Code': row['GenerationUnitCode'], # from assignment file
            'Type': row['GenerationUnitType'], # from assignment file
            'Capacity': capacity_instance, # from assignment file, scaled
            'Hub_height': df_filtered.at[current_index, "Hub height"], # from The Wind Power file
            'Commissioning_date': df_filtered.at[current_index, "Commissioning date"], # from The Wind Power file
            'Number_of_turbines': int(df_filtered.at[current_index, "Number of turbines"]), # from The Wind Power file (value only valid for latest WPPs)
            'Turbine': df_filtered.at[current_index, "Turbine"], # from The Wind Power file
            'Latitude': df_filtered.at[current_index, "Latitude"], # from The Wind Power file
            'Longitude': df_filtered.at[current_index, "Longitude"], # from The Wind Power file
            'Production': production_array_instance # from JSON file, scaled
        }

        production_data.append(row_data)

print("number WPPs after preprocessing", len(production_data))

# JSON-Datei speichern
with open("data/WPPs+production_new.json", 'w', encoding='utf-8') as json_file:
    json.dump(production_data, json_file, ensure_ascii=False, indent=4)

print(f"Zusammengeführte JSON-Datei wurde erfolgreich gespeichert unter: {output_file}")

# Convert the list to a DataFrame
df_production_data = pd.DataFrame(production_data)

# Save the DataFrame to an Excel file
df_production_data.to_excel("data/WPPs+production_new.xlsx", index=False)