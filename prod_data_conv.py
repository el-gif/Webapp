import pandas as pd

# Datei einlesen
file_path = r"C:\Users\alexa\Documents\Webapp\data\production_history\2015_01_ActualGenerationOutputPerGenerationUnit_16.1.A_r2.1.csv"
data = pd.read_csv(file_path, sep='\t')

# Filtere nach GenerationUnitType == 'Wind Onshore' oder 'Wind Offshore'
filtered_data = data[(data['GenerationUnitType'] == 'Wind Onshore ') | (data['GenerationUnitType'] == 'Wind Offshore ')]

# Wichtige Spalten identifizieren
unique_windfarms = filtered_data[['GenerationUnitName', 'GenerationUnitCode', 'GenerationUnitType', 'AreaDisplayName', 'MapCode', 'AreaTypeCode']].drop_duplicates()

# Listen für die Produktion zu jeder Stunde hinzufügen
production_data = []
for _, row in unique_windfarms.iterrows():
    windfarm_data = filtered_data[filtered_data['GenerationUnitName'] == row['GenerationUnitName']]
    production_values = windfarm_data['ActualGenerationOutput(MW)'].tolist()
    row_data = {
        'GenerationUnitName': row['GenerationUnitName'],
        'GenerationUnitCode': row['GenerationUnitCode'],
        'GenerationUnitType': row['GenerationUnitType'],
        'AreaDisplayName': row['AreaDisplayName'],
        'MapCode': row['MapCode'],
        'AreaTypeCode': row['AreaTypeCode'],
        'Production (MW)': [production_values]  # Hier die Produktion als Liste speichern
    }
    production_data.append(row_data)

# DataFrame erstellen und in Excel speichern
output_df = pd.DataFrame(production_data)
output_path = r"C:\Users\alexa\Documents\Webapp\data\production_summary.xlsx"
output_df.to_excel(output_path, index=False)

print("Excel-Datei wurde erfolgreich erstellt:", output_path)
