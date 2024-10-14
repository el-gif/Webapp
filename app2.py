from ipyleaflet import Map, Heatmap, LegendControl
from ipyleaflet.velocity import Velocity
from ipywidgets import Layout
from shiny.express import ui, input
from shinywidgets import render_widget, output_widget
from shiny import reactive
import pickle
import numpy as np
import itertools
import numpy as np
from scipy.interpolate import griddata
import xarray as xr
from branca.colormap import linear

# Beispiel-Windgeschwindigkeitsspanne (min und max Windgeschwindigkeit in m/s)
min_wind_speed = 0
max_wind_speed = 30

# Erstelle eine Colormap, die zwischen min_wind_speed und max_wind_speed skaliert
colormap = linear.Paired_06.scale(min_wind_speed, max_wind_speed)

# Datei laden und als xarray-Dataset öffnen
file_path = "data_europe.grib2"
ds = xr.open_dataset(file_path, engine='cfgrib')

# Bereich für Europa definieren
lat_min, lat_max = 35,72 # Breitengradbereich für Europa: 35, 72
lon_min, lon_max = -25, 45 # Längengradbereich für Europa: -25, 45

# Extrahiere die Windkomponenten für den aktuellen Zeitschritt
lats = ds['latitude'].values
lons = ds['longitude'].values

# Filtere die Breitengrade (latitude) und Längengrade (longitude) auf den gewünschten Bereich für Europa
lat_indices = np.where((lats >= lat_min) & (lats <= lat_max))[0]
lon_indices = np.where((lons >= lon_min) & (lons <= lon_max))[0]

# Trunkiere die Längen- und Breitengrade auf den europäischen Bereich
lats_selection = lats[lat_indices]
lons_selection = lons[lon_indices]

# Iteriere über alle Zeitschritte (steps) im Dataset
u_selection = []
v_selection = []

for i in range(len(ds['step'])):

    # Extrahiere die Windkomponenten für den aktuellen Zeitschritt
    u = ds['u100'].isel(step=i).values
    v = ds['v100'].isel(step=i).values
 
    # Extrahiere die Windgeschwindigkeitsdaten für Europa
    u_selection.append(u[np.ix_(lat_indices, lon_indices)])
    v_selection.append(v[np.ix_(lat_indices, lon_indices)])

with open("step_selection.pkl", "rb") as f:
    data = pickle.load(f)

step_selection = list(data['step_selection']) # conversion dict -> list
initial_step = step_selection[0]
final_step = step_selection[-1]

# Interpolation to the following timely resolution for forecast in hours
step_size = 1

# Function for temporal interpolation
def interpolation(list):
    list_interpol = []
    for i in range(len(list) - 1):
        step_start = step_selection[i]
        step_end = step_selection[i + 1]
        list_start = list[i]
        list_end = list[i + 1]
        list_interpol.append(list_start)
        interval = step_end - step_start
        for step in np.arange(step_start + step_size, step_end, step_size):
            factor = (step - step_start) / interval
            interpolated_value = (1 - factor) * np.array(list_start) + factor * np.array(list_end)
            list_interpol.append(interpolated_value)
    list_interpol.append(list[-1])
    return list_interpol

# Interpolation
u_selection_interpol = interpolation(u_selection)
v_selection_interpol = interpolation(v_selection)

total_selection_interpol = []
factor = 2 # interpolation for colormap, should be > 1

for i in range(len(u_selection_interpol)):

    # Berechne die Windgeschwindigkeit für den europäischen Bereich
    total_selection_interpol.append(np.sqrt(u_selection_interpol[i]**2 + v_selection_interpol[i]**2))

def remove_all_layers(m):
    for i in range(1, len(m.layers)):
        m.remove_layer(m.layers[1])
        
ui.h2("An ipyleaflet Map", style="display:none;"),  # Entferne oder verstecke die Überschrift, wenn nicht notwendig

# Verwende CSS, um die Karte auf volle Bildschirmgröße zu setzen
ui.tags.style("""
    html, body {
        margin: 0;
        padding: 0;
        height: 100%;
        width: 100%;
        overflow: hidden;
    }
"""),

# Slider für die Zeitschritte (auf der Karte überlagert)
ui.tags.div(
    ui.input_slider("time_step", "Time Step (hours)", min=initial_step, max=final_step, step=step_size, value=initial_step),
    style="position: absolute; top: 20px; left: 20px; background-color: white; padding: 10px; z-index: 1000;"
)

from ipyleaflet import LegendControl

def create_legend_dict(colormap, steps=10):
    """Erstellt ein Legenden-Dictionary aus einer kontinuierlichen Colormap."""
    step_values = np.linspace(min_wind_speed, max_wind_speed, steps)
    legend_dict = {f"{round(v, 2)} m/s": colormap(v) for v in step_values}
    return legend_dict

def add_continuous_legend(m, colormap):
    # Erstelle die Legende als Dictionary mit diskreten Schritten
    legend_dict = create_legend_dict(colormap, steps=10)
    legend_control = LegendControl(legend=legend_dict, position="bottomright")
    m.add_control(legend_control)



@render_widget  # server logic is embedded directly using reactive.effect and render_widget decorators, no ui necessary to display element for map
def map():
    m = Map(center=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
            zoom=3,
            tiles=f'https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{{z}}/{{x}}/{{y}}?access_token=pk.eyJ1IjoiZWwtZ2lmIiwiYSI6ImNtMXQyYWdsYzAwMGUycXFzdmY2eDFnaWMifQ.yirQoMK5TCdmZZUFUNXxwA',
            attr='Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="https://www.mapbox.com/">Mapbox</a>',
            layout=Layout(width='100%', height='100vh')
        )
    return m


@reactive.effect
def add_layer():
    m = map.widget
    remove_all_layers(m)

    # Slider time step
    time_step = input.time_step()
    step_index = int((time_step - initial_step) / step_size)

    # Erzeuge das kartesische Produkt, um alle Kombinationen von Breiten- und Längengraden zu erhalten
    grid_points = list(itertools.product(lats_selection, lons_selection))

    # Entpacke die Koordinaten in separate Listen
    latitudes, longitudes = zip(*grid_points)

    # Wende die Colormap auf die Windgeschwindigkeitswerte an (Farben)
    heatmap_data = []
    for lat, lon, wind_speed in zip(latitudes, longitudes, total_selection_interpol[step_index].flatten()):
        # Füge die Position und die Windgeschwindigkeit (Farbe) zur Heatmap hinzu
        heatmap_data.append((lat, lon, wind_speed))  # Heatmap arbeitet mit (lat, lon, intensity)

    # Heatmap erstellen
    heatmap = Heatmap(locations=heatmap_data, radius=5, blur=5, max_zoom=10)

    # Füge die Heatmap zur Karte hinzu
    m.add_layer(heatmap)

    # Füge die kontinuierliche Legende hinzu
    add_continuous_legend(m, colormap)

    display_options = {
        'velocityType': 'European Wind',
        'displayPosition': 'bottomleft',
        'displayEmptyString': 'No wind data'
    }

    wind_layer = Velocity(
        data={
            'u': (['lat', 'lon'], u_selection_interpol[step_index]),
            'v': (['lat', 'lon'], v_selection_interpol[step_index]),
            'lat': lats_selection,
            'lon': lons_selection,
        },
        zonal_speed='u',  # Zonal (east-west) wind component
        meridional_speed='v',  # Meridional (north-south) wind component
        latitude_dimension='lat',
        longitude_dimension='lon',
        velocity_scale=0.01,  # Scale of the wind arrows
        max_velocity=20,  # Maximum velocity for scaling the arrows
        display_options=display_options  # Use the display options we defined
    )


    m.add_layer(wind_layer)  # Add wind velocity layer to the map




# to run: shiny run app2.py