from shiny import App, ui
from shinywidgets import output_widget, render_widget
import pandas as pd
from ipyleaflet import Map, Marker, MarkerCluster, AwesomeIcon
from ipywidgets import Layout, HTML

# Daten einlesen
file1 = "C:\\Users\\alexa\\Documents\\Webapp\\data\\WPPs\\Global-Wind-Power-Tracker-June-2024.xlsx"
file2 = "C:\\Users\\alexa\\Documents\\Webapp\\data\\WPPs\\Open Power System Data renewable_power_plants_FR.csv"

data1 = pd.read_excel(file1, sheet_name='Data')
data2 = pd.read_csv(file2, low_memory=False)

# Daten für Frankreich filtern und fehlende Koordinaten entfernen
data1_france = data1[data1['Country/Area'] == 'France'].dropna(subset=['Latitude', 'Longitude', 'Project Name'])
data2 = data2[data2['energy_source_level_2'] == 'Wind'].dropna(subset=['lat', 'lon', 'site_name'])

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
        m = Map(center=(46.603354, 1.888334), zoom=6, layout=Layout(width='100%', height='100vh'), scroll_wheel_zoom=True)  # Zentriert auf Frankreich

        # Farbe für die Marker der beiden Layer
        icon_layer1 = AwesomeIcon(name="info-sign", marker_color="blue", icon_color="white")
        icon_layer2 = AwesomeIcon(name="info-sign", marker_color="green", icon_color="white")

        # Layer 1 für "Global-Wind-Power-Tracker-June-2024.xlsx" (blau)
        markers1 = [Marker(location=(row['Latitude'], row['Longitude']),
                           popup=HTML(value=row['Project Name']),
                           icon=icon_layer1) for _, row in data1_france.iterrows()]
        marker_cluster1 = MarkerCluster(markers=markers1)
        m.add_layer(marker_cluster1)

        # Layer 2 für "Open Power System Data renewable_power_plants_FR.csv" (grün)
        markers2 = [Marker(location=(row['lat'], row['lon']),
                           popup=HTML(value=row['site_name']),
                           icon=icon_layer2) for _, row in data2.iterrows()]
        marker_cluster2 = MarkerCluster(markers=markers2)
        m.add_layer(marker_cluster2)

        return m

# App erstellen
app = App(app_ui, server)
app.run()
