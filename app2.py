from ipyleaflet import Map, Heatmap
from shiny.express import ui, input
from shinywidgets import render_widget
from shiny import reactive
import pickle
import numpy as np
import itertools

# Load the layer data
with open("wind_layers.pkl", "rb") as f:
    data_to_save = pickle.load(f)

layers = data_to_save['layers']
lat_min = data_to_save['lat_min']
lat_max = data_to_save['lat_max']
lon_min = data_to_save['lon_min']
lon_max = data_to_save['lon_max']
lats_europe = data_to_save['lats_europe']
lons_europe = data_to_save['lons_europe']
wind_speed = data_to_save['wind_speed']
step_selection = np.array(data_to_save['step_selection'])

step_size = 1
initial_step = step_selection.min()

# # Interpolate layers between known steps
# all_layers = []
# for i in range(len(step_selection) - 1):
#     step_start = step_selection[i]
#     step_end = step_selection[i + 1]
#     layer_start = layers[i]
#     layer_end = layers[i + 1]
#     all_layers.append(layer_start)
#     interval = step_end - step_start
#     for step in np.arange(step_start + step_size, step_end, step_size):
#         factor = (step - step_start) / interval
#         interpolated_image = (1 - factor) * np.array(layer_start[1]) + factor * np.array(layer_end[1])
#         interpolated_layer = (f"Interpoliert {step:.2f}", interpolated_image)
#         all_layers.append(interpolated_layer)
# all_layers.append(layers[-1])

all_wind_speeds = []
for i in range(len(wind_speed) - 1):
    step_start = step_selection[i]
    step_end = step_selection[i + 1]
    wind_speed_start = wind_speed[i]
    wind_speed_end = wind_speed[i + 1]
    all_wind_speeds.append(wind_speed_start)
    interval = step_end - step_start
    for step in np.arange(step_start + step_size, step_end, step_size):
        factor = (step - step_start) / interval
        interpolated_wind_speed = (1 - factor) * np.array(wind_speed_start) + factor * np.array(wind_speed_end)
        all_wind_speeds.append(interpolated_wind_speed)
all_wind_speeds.append(wind_speed[-1])

# Initial Zoom-Level and Map Center
zoom_level = 6
map_center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

def remove_all_layers(m):
    for i in range(1, len(m.layers)):
        m.remove_layer(m.layers[1])
        
ui.h2("An ipyleaflet Map"),

ui.input_slider("time_step", "Time Step (hours)", min=step_selection.min(), max=step_selection.max(), step=step_size, value=initial_step)



# ui.output_text("selected_step")

# @render.text
# def selected_step():
#     return f"Zeitschritt: {input.time_step():.2f} Stunden"

@render_widget  # server logic is embedded directly using reactive.effect and render_widget decorators, no ui necessary to display element for map
def map():
    center = (51.1657, 10.4515)
    m = Map(center=map_center,
            zoom=zoom_level,
            tiles=f'https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token=pk.eyJ1IjoiZWwtZ2lmIiwiYSI6ImNtMXQyYWdsYzAwMGUycXFzdmY2eDFnaWMifQ.yirQoMK5TCdmZZUFUNXxwA',
            attr='Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="https://www.mapbox.com/">Mapbox</a>'
        )

    return m



@reactive.effect
def add_layer():
    m = map.widget
    remove_all_layers(m)

    # Slider time step
    time_step = input.time_step()
    step_index = int((time_step - step_selection.min()) / step_size)

    # Erzeuge das kartesische Produkt, um alle Kombinationen von Breiten- und Längengraden zu erhalten
    grid_points = list(itertools.product(lats_europe, lons_europe))

    # Entpacke die Koordinaten in separate Listen
    latitudes, longitudes = zip(*grid_points)

    # Kombiniere die Punkte in ein Format, das die Heatmap verwenden kann
    heatmap_data = list(zip(latitudes, longitudes, all_wind_speeds[step_index].flatten()))

    # Heatmap erstellen
    heatmap = Heatmap(locations=heatmap_data, radius=10, blur=30, max_zoom=10)

    # Füge die Heatmap zur Karte hinzu
    m.add_layer(heatmap)


# to run: shiny run app2.py