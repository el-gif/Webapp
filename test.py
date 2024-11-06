from shiny import App, ui
from shinywidgets import output_widget, render_widget
from ipyleaflet import Map, Marker, MarkerCluster, AwesomeIcon
from ipywidgets import Layout, HTML
import xml.etree.ElementTree as ET

# Pfad zur XML-Datei
file_path = "C:\\Users\\alexa\\Documents\\Webapp\\data\\WPPs\\MaStR_EinheitenWind.xml"

# XML-Daten einlesen und filtern
tree = ET.parse(file_path)
root = tree.getroot()

# Filtere die Anlagen mit einer Bruttoleistung über 1000 kW
anlagen_data = []
for einheit in root.findall(".//EinheitWind"):
    bruttoleistung = float(einheit.find("Bruttoleistung").text)
    if bruttoleistung > 1000:  # nur Anlagen über 1000 kW
        lat = float(einheit.find("Breitengrad").text)
        lon = float(einheit.find("Laengengrad").text)
        name = einheit.find("NameStromerzeugungseinheit").text
        anlagen_data.append({"lat": lat, "lon": lon, "name": name, "leistung": bruttoleistung})

# Shiny-App-UI
app_ui = ui.page_fluid(
    output_widget("map")
)

# Serverlogik
def server(input, output, session):
    @output
    @render_widget
    def map():
        # Grundkarte mit ipyleaflet erstellen
        m = Map(center=(51.0, 10.0), zoom=6, layout=Layout(width='100%', height='100vh'), scroll_wheel_zoom=True)  # Zentriert auf Deutschland

        # Erstelle MarkerCluster für gefilterte Anlagen
        markers = [
            Marker(
                location=(anlage['lat'], anlage['lon']),
                popup=HTML(value=f"<strong>{anlage['name']}</strong><br>Leistung: {anlage['leistung']} kW")
            )
            for anlage in anlagen_data
        ]
        marker_cluster = MarkerCluster(markers=markers)
        m.add_layer(marker_cluster)

        return m

# App erstellen
app = App(app_ui, server)
app.run()
