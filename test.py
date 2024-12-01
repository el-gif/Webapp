import pandas as pd

# Laden der Daten
df_wind_power = pd.read_parquet("data/WPPs/The_Wind_Power.parquet")
df_assignment = pd.read_excel("data/Assignment.xlsx")

import math

counter = 0

# Funktion zum Ersetzen von NaN-Werten
def replace_nan(data):
    for i in range(len(data)):
        if math.isnan(data[i]):  # Überprüfen, ob der Wert NaN ist
            global counter
            counter += 1
            data[i] = data[i - 1] if i > 0 else 0
    return data

# Filtere nur Zeilen, bei denen "ID_The-Wind-Power" nicht "not found" ist
df_assignment = df_assignment[df_assignment["ID_The-Wind-Power"] != "not found"]

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
actual_ids = set(df_filtered['ID'])
suspended_ids = valid_ids - actual_ids

print("number potential WPPs:", len(valid_ids))
print("number actual WPPs:", len(actual_ids))
print("number suspended WPPs (no name, location, capacity or status not in operation):", len(suspended_ids))

# Füge Spalten für Produktion von 2015_01 bis 2024_10 hinzu
new_columns = {f"{year}_{month:02d}": [[] for _ in range(len(df_filtered))]
            for year in range(2015, 2025) for month in range(1, 13)
            if f"{year}_{month:02d}" in df_assignment.columns}

# Füge die neuen Spalten zum DataFrame hinzu
df_filtered = pd.concat([df_filtered, pd.DataFrame(new_columns, index=df_filtered.index)], axis=1)

# Gehe durch jede Zeile der Assignment-Datei und füge Produktionsdaten hinzu
for index, row in df_assignment.iterrows():
    
    ids_in_row = extract_ids(row["ID_The-Wind-Power"])
    first_id = ids_in_row[0]

    if first_id in suspended_ids:
        continue # jump to next iteration, because following line would fail for suspended_ids

    current_index = df_filtered.loc[df_filtered['ID'] == first_id].index[0]

    # add production values for each month, only requires the first ID
    for year in range(2015, 2025):
        for month in range(1, 13):
            column_name = f"{year}_{month:02d}"
            if column_name in df_assignment.columns: # neglect 2024_11 and 2024_12
                production_month = row[column_name]
                if isinstance(production_month, str): # type(value) = str, meaning value = production values
                    production_month = production_month.replace("nan", "float('nan')")
                    production_month = eval(production_month) # [[]] <-- "[[]]"
                    production_month = production_month[0] # [] <-- [[]]
                    # value = replace_nan(value)
                    existing_production = df_filtered.at[current_index, column_name]
                    
                    if existing_production == []: # no production values in cell for this month
                        df_filtered.at[current_index, column_name] = production_month
                    else: # several production values to be added to one WPP
                        combined_production = [a + b for a, b in zip(existing_production, production_month)]
                        df_filtered.at[current_index, column_name] = combined_production

    # add capacities of WPPs, if several are assigned to one row in the assignment table
    if first_id in actual_ids and len(ids_in_row) > 1: # only treat every id once here, because rows are discarded
        total_power = 0
        for id in ids_in_row:
            total_power += df_filtered.loc[df_filtered['ID'] == id, "Total power"].item() # add power
            if id != first_id:
                df_filtered = df_filtered[df_filtered['ID'] != id] # delete from dataframe
        df_filtered.loc[df_filtered['ID'] == first_id, "Total power"] = total_power
    
    # keep track of treated IDs to not try to delete rows twice 
    for id in ids_in_row:
        if id in actual_ids:
            actual_ids.discard(id)

actual_cluster_ids = set(df_filtered['ID'])
print("number WPPs after clustering", len(actual_cluster_ids))
print(f"{counter} nan values have been estimated")
df_filtered.to_excel("data/WPPs+production_history.xlsx", index=False)