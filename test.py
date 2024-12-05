import pandas as pd
import json

# Laden der Daten
df_wind_power = pd.read_parquet("data/WPPs/The_Wind_Power.parquet")
df_assignment = pd.read_excel("data/Assignment.xlsx", sheet_name="Sheet1")
with open(r"C:\Users\alexa\Documents\Webapp\data\production_history\production_summary_all.json", "r") as file:
    df_json = json.load(file)
    
output_file = "data/WPPs+production.json"

# Filtere nur Zeilen, bei denen "ID_The-Wind-Power" nicht "not found" ist
df_assignment = df_assignment[df_assignment["ID_The-Wind-Power"] != "not found"]

# set wirh unique generation unit codes
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
actual_ids = set(df_filtered['ID'])
suspended_ids = valid_ids - actual_ids

print("number potential WPPs:", len(valid_ids))
print("number actual WPPs:", len(actual_ids))
print("number suspended WPPs (no name, location, capacity or status not in operation):", len(suspended_ids))

production_data = [] # neues JSON-File mit Produktionsdaten für die WPPs
temporal_wpps = [] # WPPs, die temporär gespeichert werden, um sie später zu aktualisieren

# Gehe durch jede Zeile der Assignment-Datei und füge Produktionsdaten hinzu
for _, row in df_assignment.iterrows():
    
    ids_in_row = extract_ids(row["ID_The-Wind-Power"])
    first_id = ids_in_row[0] # dismiss other ids in the same row, because the capacity of the WPP is not taken from the wind power database anyway and other statistics should be the same for all indices

    if first_id in suspended_ids:
        continue # jump to next iteration, because following line would fail for suspended_ids

    production_array = df_json[row['JSON-ID']]['Production']
    capacity = row['GenerationUnitInstalledCapacity(MW)']

    if first_id not in actual_ids: # several lines in assignment files for one WPP in The Wind Power file
        if row['GenerationUnitCode'] not in generation_unit_code_set: # another row with the same generation unit code as a previous row --> create new WPP although its first_id is identical, because the capacity differs
            pass # continue at current_index = ...
        else: # add production data to existing WPP
            pass
            for _, wpp in enumerate(production_data):
                if wpp['ID_The-Wind-Power'] == first_id:

                    existing_production = wpp['Production']

                    # Vergleiche Zeitstempel und addiere nur bei Übereinstimmung
                    i, j = 0, 0  # Zwei Zeiger für existing_production und production_array
                    updated_production = []

                    while i < len(existing_production) and j < len(production_array):
                        time, existing_value = existing_production[i]
                        time_comp, new_value = production_array[j]

                        if time == time_comp:
                            updated_production.append([time, existing_value + new_value])
                            i += 1
                            j += 1
                        elif time < time_comp:
                            i += 1
                        else:
                            j += 1

                    if updated_production != []:
                        wpp['Production'] = updated_production # update production data (# Ergebnisliste enthält nur Einträge mit übereinstimmenden Zeitstempeln)
                        wpp['Capacity'] = wpp['Capacity'] + capacity # update capacity
                        temporal_wpps.append(wpp)
            continue # don't add another time to the production data
    else: # after wpps' production has been changed, treat temporal_wpps. Only possible now, because some wpps were needed multiple times
        if len(temporal_wpps) > 0:
            for wpp_new in temporal_wpps:
                # if available, delete the wpp from production data (recognised by GenerationUnitCode and GenerationUnitInstalledCapacity(MW))
                production_data = [wpp for wpp in production_data if not (wpp['Code'] == wpp_new['Code'] and wpp['Capacity'] == wpp_new['Capacity'])]
                production_data.append(wpp_new)
            temporal_wpps = []

    current_index = df_filtered.loc[df_filtered['ID'] == first_id].index[0]

    # Daten für das Windkraftwerk hinzufügen
    row_data = {
        'Name': row['GenerationUnitName'], # from assignment file
        'ID_The-Wind-Power': first_id, # from assignment file
        'JSON-ID': row['JSON-ID'], # from assignment file
        'Code': row['GenerationUnitCode'], # from assignment file
        'Type': row['GenerationUnitType'], # from assignment file
        'Capacity': capacity, # from assignment file
        'Hub_height': df_filtered.at[current_index, "Hub height"], # from The Wind Power file
        'Commission_date': df_filtered.at[current_index, "Commissioning date"], # from The Wind Power file
        'Number_of_turbines': int(df_filtered.at[current_index, "Number of turbines"]), # from The Wind Power file (value only valid for latest WPPs)
        'Turbine': df_filtered.at[current_index, "Turbine"], # from The Wind Power file
        'Production': production_array # from JSON file
    }

    production_data.append(row_data)

    # keep track of treated generation unit codes
    generation_unit_code_set.discard(row['GenerationUnitCode'])

    # keep track of treated IDs to not try deleting rows twice 
    for id in ids_in_row:
        if id in actual_ids:
            actual_ids.discard(id)

print("number WPPs after clustering", len(production_data))

# JSON-Datei speichern
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(production_data, json_file, ensure_ascii=False, indent=4)

print(f"Zusammengeführte JSON-Datei wurde erfolgreich gespeichert unter: {output_file}")

# Convert the list to a DataFrame
df_production_data = pd.DataFrame(production_data)

# Save the DataFrame to an Excel file
df_production_data.to_excel("data/WPPs+production.xlsx", index=False)