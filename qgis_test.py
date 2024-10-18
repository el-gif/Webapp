import sys
from qgis.core import QgsApplication, QgsVectorLayer

# Setze den Pfad zu deiner QGIS-Installation
sys.path.append("C:/OSGeo4W64/apps/qgis/python")  # Pfad zu QGIS Python

# Initialisiere die QGIS-Anwendung
app = QgsApplication([], False)
app.initQgis()

# Lade ein Vektorlayer
layer_path = "path/to/your/layer.shp"  # Pfad zu deinem Shapefile
layer = QgsVectorLayer(layer_path, "Layername", "ogr")

if not layer.isValid():
    print("Layer konnte nicht geladen werden!")
else:
    print(f"Layer '{layer.name()}' wurde erfolgreich geladen.")
    print(f"Anzahl der Merkmale: {layer.featureCount()}")

# Schließe die QGIS-Anwendung
app.exitQgis()
