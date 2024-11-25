import pandas as pd
import os

# Basisverzeichnisse
input_dir = r"C:\Users\alexa\Documents\Webapp\data\production_history\raw"
output_dir = r"C:\Users\alexa\Documents\Webapp\data\production_history\processed"

# Liste der Monate von 2015_06 bis 2024_10 generieren
months = pd.date_range(start="2021-06", end="2024-10", freq="M").strftime("%Y_%m").tolist()

# For-Schleife für jede Datei
for month in months:
    # Dateipfad erstellen
    input_file = os.path.join(input_dir, f"{month}_ActualGenerationOutputPerGenerationUnit_16.1.A_r2.1.csv")
    output_file = os.path.join(output_dir, f"production_summary_{month}.xlsx")

    # Überprüfen, ob die Datei existiert
    if not os.path.exists(input_file):
        print(f"Datei nicht gefunden: {input_file}")
        continue  # Überspringt diese Iteration, wenn die Datei nicht existiert

    # Datei einlesen
    print(f"Bearbeite Datei: {input_file}")
    data = pd.read_csv(input_file, sep='\t')

    # Filtere nach GenerationUnitType == 'Wind Onshore' oder 'Wind Offshore'
    filtered_data = data[(data['GenerationUnitType'] == 'Wind Onshore ') | (data['GenerationUnitType'] == 'Wind Offshore ')]

    # Wichtige Spalten identifizieren
    unique_windfarms = filtered_data[['GenerationUnitName', 'GenerationUnitCode', 'GenerationUnitType', 'AreaDisplayName', 'MapCode', 'AreaTypeCode', 'GenerationUnitInstalledCapacity(MW)']].drop_duplicates()

    # Listen für die Produktion zu jeder Stunde hinzufügen
    production_data = []
    for _, row in unique_windfarms.iterrows():
        windfarm_data = filtered_data[filtered_data['GenerationUnitName'] == row['GenerationUnitName']]
        production_values = windfarm_data['ActualGenerationOutput(MW)'].tolist()
        row_data = {
            'GenerationUnitName': row['GenerationUnitName'],
            'GenerationUnitCode': row['GenerationUnitCode'],
            'GenerationUnitType': row['GenerationUnitType'],
            'GenerationUnitInstalledCapacity(MW)': row['GenerationUnitInstalledCapacity(MW)'],
            'AreaDisplayName': row['AreaDisplayName'],
            'MapCode': row['MapCode'],
            'AreaTypeCode': row['AreaTypeCode'],
            'Production (MW)': [production_values]  # Hier die Produktion als Liste speichern
        }
        production_data.append(row_data)

    # DataFrame erstellen und in Excel speichern
    output_df = pd.DataFrame(production_data)
    output_df.to_excel(output_file, index=False)

    print(f"Excel-Datei wurde erfolgreich erstellt: {output_file}")
