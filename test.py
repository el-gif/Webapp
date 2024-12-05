import pandas as pd
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Basisverzeichnisse
input_dir = r"E:\MA_data\raw production history ENTSO-E"
output_dir = r"C:\Users\alexa\Documents\Webapp\data\production_history\processed_new"

# Liste der Monate von 2015-01 bis 2024-10 generieren
months = pd.date_range(start="2015-01", end="2015-01", freq="MS").strftime("%Y_%m").tolist()

# For-Schleife für jede Datei
for month in months:
    # Dateipfad erstellen
    input_file = os.path.join(input_dir, f"{month}_ActualGenerationOutputPerGenerationUnit_16.1.A_r2.1.csv")
    output_file = os.path.join(output_dir, f"production_summary_{month}.xlsx")

    # Überprüfen, ob die Datei existiert
    if not os.path.exists(input_file):
        print(f"Datei nicht gefunden: {input_file}")
        continue

    # Datei einlesen
    print(f"Bearbeite Datei: {input_file}")
    data = pd.read_csv(input_file, sep='\t')

    # Filtere nach GenerationUnitType == 'Wind Onshore' oder 'Wind Offshore'
    filtered_data = data[(data['GenerationUnitType'] == 'Wind Onshore ') | (data['GenerationUnitType'] == 'Wind Offshore ')]

    # Konvertiere 'DateTime (UTC)' in datetime
    filtered_data.loc[:, 'DateTime (UTC)'] = pd.to_datetime(filtered_data['DateTime (UTC)'])

    # Wichtige Spalten identifizieren, 'AreaCode', 'AreaDisplayName', 'AreaTypeCode' and 'MapCode' of identical WPPs may differ --> use at least one of them as a criterion to identify unique windfarms, and sort out the duplicates manually, because otherwise, the production data are appended twice to the same wind farm
    unique_windfarms = filtered_data[['GenerationUnitName', 'GenerationUnitCode', 'GenerationUnitType', 'GenerationUnitInstalledCapacity(MW)', 'AreaCode']].drop_duplicates()

    # Listen für die Produktion zu jeder Stunde hinzufügen
    production_data = []
    counter = 0
    for _, row in unique_windfarms.iterrows():
        counter += 1
        # Filtern der Daten für das aktuelle Windkraftwerk
        windfarm_data = filtered_data[
            (filtered_data['GenerationUnitName'] == row['GenerationUnitName']) &
            (filtered_data['AreaCode'] == row['AreaCode']) # important to avoid adding to a wind farm production data of all its duplicates
        ]

        # Erstelle Liste mit Sublisten mit Zeit und Produktion. 2D-Array nicht sinnvoll, da keine Kommas eingefügt werden und Zeilenumbrüche zwischen den Sublisten entstehen --> Schwiereigkeiten beim erneuten Einlesen
        production_array = [
            [time, production]
            for time, production in zip(
                windfarm_data['DateTime (UTC)'],
                windfarm_data['ActualGenerationOutput(MW)']
            )
            if pd.notna(production)  # Überspringe fehlende Werte
        ]

        # Daten für das Windkraftwerk hinzufügen
        row_data = {
            'GenerationUnitName': row['GenerationUnitName'],
            'GenerationUnitCode': row['GenerationUnitCode'],
            'GenerationUnitType': row['GenerationUnitType'],
            'GenerationUnitInstalledCapacity(MW)': row['GenerationUnitInstalledCapacity(MW)'],
            'Production': production_array
        }
        production_data.append(row_data)

    # DataFrame erstellen und in Excel speichern
    output_df = pd.DataFrame(production_data)
    output_df.to_excel(output_file, index=False)

    print(f"Excel-Datei wurde erfolgreich erstellt: {output_file}")