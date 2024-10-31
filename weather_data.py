import os
import time
from ecmwf.opendata import Client

# Pfad zu den gespeicherten Wetterdaten
data_file_path = "data_europe.grib2"
# Festgelegtes Zeitintervall für Aktualisierungen (in Sekunden)
time_threshold = 10 * 3600  # 10 Stunden in Sekunden
overwrite = 0 # force overwriting of data

# Funktion, um das Alter der Datei zu überprüfen
def is_data_stale(file_path, time_threshold):
    if not os.path.exists(file_path):
        # Datei existiert nicht, Daten müssen abgerufen werden
        return True
    # Alter der Datei ermitteln (in Sekunden seit der letzten Änderung)
    file_age = time.time() - os.path.getmtime(file_path)
    # Überprüfen, ob die Datei älter als der festgelegte Schwellenwert ist
    return file_age > time_threshold

# ECMWF-Client initialisieren
client = Client(
    source="ecmwf",
    model="ifs",
    resol="0p25"
)

# all available steps up to 7 days (168 hours)
step_selection = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60,
     63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120,
     123, 126, 129, 132, 135, 138, 141, 144, 150, 156, 162, 168]
# all available steps
# step_selection = [
#     0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60,
#     63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117, 120,
#     123, 126, 129, 132, 135, 138, 141, 144, 150, 156, 162, 168, 174, 180, 186, 192, 198, 204,
#     210, 216, 222, 228, 234, 240
# ]

if is_data_stale(data_file_path, time_threshold) or overwrite:
    print("Daten sind veraltet, existieren nicht oder sollen überschrieben werden. Abruf neuer Daten.")
    # Abrufen der API-Daten, da sie entweder fehlen oder älter als 10 Stunden sind
    result = client.retrieve( # retrieve data worldwide, because no area tag available (https://github.com/ecmwf/ecmwf-opendata, https://www.ecmwf.int/en/forecasts/datasets/open-data)
        type="fc",  
        param=["100v", "100u"],  # U- und V-Komponenten der Windgeschwindigkeit
        target=data_file_path,
        time=0,  # Vorhersagezeit (Modelllauf um 00z)
        step=step_selection
    )
    print("Neue Daten wurden erfolgreich abgerufen und gespeichert.")
else:
    print("Daten sind aktuell und werden verwendet.")
