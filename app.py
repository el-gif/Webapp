from shiny import App, ui, render
import folium
import pickle
import numpy as np

# Lade die Layer-Daten
with open("wind_layers.pkl", "rb") as f:
    data_to_save = pickle.load(f)

layers = data_to_save['layers']
lat_min = data_to_save['lat_min']
lat_max = data_to_save['lat_max']
lon_min = data_to_save['lon_min']
lon_max = data_to_save['lon_max']
step_selection = np.array(data_to_save['step_selection'])

step_size = 1

# Original Layer-Werte entsprechen den diskreten Steps
original_layer_values = np.arange(len(layers))

# Erstelle eine Liste, die die Original- und interpolierten Layer enthalten wird
all_layers = []

# Interpoliere Layer-Daten zwischen den bekannten Schritten
for i in range(len(step_selection) - 1):
    step_start = step_selection[i]
    step_end = step_selection[i + 1]
    layer_start = layers[i]
    layer_end = layers[i + 1]

    # Füge das Original-Layer hinzu (für step_start)
    all_layers.append(layer_start)

    # Berechne die Anzahl der Interpolationspunkte (zwischen 3 oder 6 Stunden)
    interval = step_end - step_start

    # Interpolation für jeden 1-Schritt zwischen step_start und step_end
    for step in np.arange(step_start + step_size, step_end, step_size):
        # Linearer Interpolationsfaktor (zwischen 0 und 1)
        factor = (step - step_start) / interval
        
        # Interpoliertes Bild erstellen (linear für Einfachheit, aber könnte quadratisch sein)
        interpolated_image = (1 - factor) * np.array(layer_start[1]) + factor * np.array(layer_end[1])
        interpolated_layer = (f"Interpoliert {step:.2f}", interpolated_image)
        
        # Füge den interpolierten Layer zu all_layers hinzu
        all_layers.append(interpolated_layer)

# Füge den letzten Original-Layer hinzu
all_layers.append(layers[-1])

# Initialisiere Zoom-Level und Kartenmittelpunkt
zoom_level = 6  # Anfangszoom
map_center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]  # Anfangsmittelpunkt (Europa)

# Define UI with CSS for absolute positioning
app_ui = ui.page_fluid(
    # Container for the map and overlay UI elements
    ui.tags.div(
        ui.output_ui("wind_map"),  # The map in the background
        ui.tags.div(
            ui.h2("Windgeschwindigkeitsvorhersage für Europa"),
            ui.input_slider("time_step", "Zeitschritt (Stunden)", min=step_selection.min(), max=step_selection.max(), step=step_size, value=step_selection.min()),
            ui.output_text("selected_step"),
            style="position:absolute; top:20px; left:20px; background-color:rgba(255, 255, 255, 0.8); padding:10px; z-index:1000;"
        ),
        style="position:relative; height:600px;"  # Make sure the container has enough height
    )
)

# Define server logic
def server(input, output, session):

    # Text output for the selected step
    @output
    @render.text
    def selected_step():
        return f"Zeitschritt: {input.time_step():.2f} Stunden"

    # Output UI for displaying the Folium map with overlay
    @output
    @render.ui
    def wind_map():
        global zoom_level, map_center

        # Erstelle die Folium-Karte, verwende den aktuellen Zoom-Level und Kartenmittelpunkt
        m = folium.Map(
            location=map_center,  # Behalte den Mittelpunkt bei
            zoom_start=zoom_level,  # Behalte den Zoom-Level bei
            tiles=f'https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token=pk.eyJ1IjoiZWwtZ2lmIiwiYSI6ImNtMXQyYWdsYzAwMGUycXFzdmY2eDFnaWMifQ.yirQoMK5TCdmZZUFUNXxwA',
            attr='Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="https://www.mapbox.com/">Mapbox</a>'
        )

        # Slider-Zeitschritt
        time_step = input.time_step()
        
        # Index basierend auf der Slider-Einstellung
        step_index = int((time_step - step_selection.min()) / step_size)
        
        # Hole den Layer entsprechend der interpolierten Werte
        layer_name, overlay_image = all_layers[step_index]

        # Füge das Bildoverlay hinzu
        folium.raster_layers.ImageOverlay(
            image=overlay_image,
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],  # Grenzen setzen (richtig herum)
            opacity=0.6,
            name=layer_name
        ).add_to(m)

        # LayerControl hinzufügen
        folium.LayerControl().add_to(m)

        # Rückgabe der Folium-Karte als HTML
        return ui.HTML(m._repr_html_())

# Initialize app
app = App(app_ui, server)

# Run the app
app.run(port=8000)
