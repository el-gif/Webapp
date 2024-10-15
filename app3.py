from shiny import App, ui
from ipywidgets import SelectionSlider, Layout, Play, VBox, jslink, Dropdown, HTML
from shinywidgets import render_widget, output_widget
from ipyleaflet import Map, Heatmap, WidgetControl, ColormapControl, LayersControl, LayerGroup, basemaps, basemap_to_tiles, FullScreenControl
from ipyleaflet.velocity import Velocity
import numpy as np
import itertools
from branca.colormap import linear
import xarray as xr

# Datei laden und als xarray-Dataset öffnen
file_path = "data_europe.grib2"
ds = xr.open_dataset(file_path, engine='cfgrib')
valid_times = ds['valid_time'].values
initial = 0

# Bereich für Europa definieren
lat_min, lat_max = 35, 72
lon_min, lon_max = -25, 45

# Initialisierung bei Durchschnittswerten
min_wind_speed = 5
max_wind_speed = 6

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
    u = ds['u100'].isel(step=i).values
    v = ds['v100'].isel(step=i).values
    u_selection.append(u[np.ix_(lat_indices, lon_indices)])
    v_selection.append(v[np.ix_(lat_indices, lon_indices)])

start_time = valid_times[0]
end_time = valid_times[-1]
total_hours = int((end_time - start_time) / np.timedelta64(1, 'h')) # no data loss, will be an integer anyway

step_size = np.timedelta64(1, 'h') # Interpolation to the following timely resolution for forecast
step_size_hours = step_size / np.timedelta64(1, 'h')

# Function for temporal interpolation
def interpolation(list):
    list_interpol = []
    for i in range(len(list) - 1):
        step_start = valid_times[i]
        step_end = valid_times[i + 1]
        list_start = list[i]
        list_end = list[i + 1]
        list_interpol.append(list_start)
        interval = step_end - step_start
        for step in np.arange(step_start + step_size, step_end, step_size):
            factor = (step - step_start) / interval
            interpolated_value = (1 - factor) * list_start + factor * list_end
            list_interpol.append(interpolated_value)
    list_interpol.append(list[-1])
    return list_interpol

# Interpolation
u_selection_interpol = interpolation(u_selection)
v_selection_interpol = interpolation(v_selection)

valid_times_interpol = []
steps = int(total_hours / step_size_hours)
for i in range(steps):
    valid_times_interpol.append(start_time + i*step_size)

total_selection_interpol = []
for i in range(len(u_selection_interpol)):
    total_selection_interpol.append(np.sqrt(u_selection_interpol[i]**2 + v_selection_interpol[i]**2))
    max_wind_speed = total_selection_interpol[i].max() if total_selection_interpol[i].max() > max_wind_speed else max_wind_speed
    min_wind_speed = total_selection_interpol[i].min() if total_selection_interpol[i].min() < min_wind_speed else min_wind_speed

# Advance preparation of HeatMap
grid_points = list(itertools.product(lats_selection, lons_selection))
latitudes, longitudes = zip(*grid_points)
heatmap_data = []
for i in range(len(total_selection_interpol)):
    heatmap_data.append([])
    for lat, lon, wind_speed in zip(latitudes, longitudes, total_selection_interpol[i].flatten()):
        heatmap_data[i].append((lat, lon, wind_speed))

# Benutzeroberfläche der Shiny App, Reihenfolge der Elemente wichtig!
app_ui = ui.page_fluid(
    output_widget("map")
)

# Serverlogik für die Shiny App
def server(input, output, session):

    # Render das Leaflet-Widget mit render_widget
    @output
    @render_widget
    def map():
        
        # Basemap-Dropdown-Optionen
        basemaps_dict = {
            "OpenStreetMap": basemaps.OpenStreetMap.Mapnik,
            "OpenTopoMap": basemaps.OpenTopoMap,
            "NASAGIBS ViirsEarthAtNight": basemaps.NASAGIBS.ViirsEarthAtNight2012
        }

        # Initiale Basemap setzen
        m = Map(center=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
                zoom=5,
                layout=Layout(width='100%', height='100vh')
            )

        # Dropdown-Menü zur Auswahl der Basemap
        dropdown = Dropdown(
            options=list(basemaps_dict.keys()),  # Basemap-Namen als Optionen
            value="OpenStreetMap",  # Standardwert
            description='Basemap'
        )
        dropdown_control = WidgetControl(widget=dropdown, position='topright')
        m.add_control(dropdown_control)

        # Aktiviere die erste Basemap
        current_basemap_layer = basemap_to_tiles(basemaps_dict[dropdown.value])
        m.add_layer(current_basemap_layer)

        # Funktion zur Aktualisierung der Basemap
        def update_basemap(change):
            nonlocal current_basemap_layer
            # Entferne die aktuelle Basemap
            m.remove_layer(current_basemap_layer)
            # Füge die neu ausgewählte Basemap hinzu
            current_basemap_layer = basemap_to_tiles(basemaps_dict[dropdown.value])
            m.add_layer(current_basemap_layer)

        # Verbinde das Dropdown-Menü mit der Update-Funktion
        dropdown.observe(update_basemap, names='value')

        # Erstelle einen Slider als WidgetControl und füge ihn zur Karte hinzu
        play = Play(min=0, max=total_hours, step=step_size_hours, value=0, interval = 500, description='Time Step') # 500 ms pro Schritt
        slider = SelectionSlider(options=valid_times_interpol, value=valid_times_interpol[0], description='Time')
        jslink((play, 'value'), (slider, 'index'))
        slider_box = VBox([play, slider])

        slider_control = WidgetControl(widget=slider_box, position='topright')
        m.add_control(slider_control)

        # Funktion zur Aktualisierung der Karte basierend auf dem Slider
        def update_map(change):

            for i in range(len(m.layers) - 1, -1, -1): # rückwärts iterieren, da Elemente gelöscht werden
                if isinstance(m.layers[i], (Heatmap, Velocity)):
                    m.remove_layer(m.layers[i])
            
            time_step = slider.value
            step_index = int((time_step - start_time) / step_size)

            heatmap = Heatmap(locations=heatmap_data[step_index], radius=5, blur=5, max_zoom=10)
            
            velocity = Velocity(
                data={
                    'u': (['lat', 'lon'], u_selection_interpol[step_index]),
                    'v': (['lat', 'lon'], v_selection_interpol[step_index]),
                    'lat': lats_selection,
                    'lon': lons_selection,
                },
                zonal_speed='u',
                meridional_speed='v',
                latitude_dimension='lat',
                longitude_dimension='lon',
                velocity_scale=0.01,
                max_velocity=20,
                display_options={
                    'velocityType': 'Wind Speed',
                    'displayPosition': 'bottomleft',
                    'displayEmptyString': 'No wind data'
                }
            )

            heatmap_layer = LayerGroup(layers=(heatmap,), name="Heatmap Layer")
            m.add(heatmap_layer)

            velocity_layer = LayerGroup(layers=(velocity,), name="Velocity Layer")
            m.add(velocity_layer)
        
        # Verbinde den Slider mit der Update-Funktion, observe function by ipywidgets instead of reactive by shiny
        slider.observe(update_map, names='value')
            
        global initial
        # Initialisiere die Karte mit dem ersten Zeitschritt
        if initial == 0:
            update_map(None)
            initial = 1

        # # Beispiel-Windgeschwindigkeitsspanne (min und max Windgeschwindigkeit in m/s)
        # min_wind_speed = 0
        # max_wind_speed = 30
        # colormap=linear.Paired_06.scale(min_wind_speed, max_wind_speed)
        # # Füge die kontinuierliche Legende hinzu
        # colormap_control = ColormapControl(caption="Windgeschwindigkeit (m/s)", colormap=colormap, value_min=min_wind_speed, value_max=max_wind_speed, position="bottomright")
        # m.add_control(colormap_control)

        layer_control = LayersControl(position='bottomleft')
        m.add_control(layer_control)

        # Erstellen einer benutzerdefinierten Legende als HTML
        colormap = linear.viridis.scale(round(min_wind_speed), round(max_wind_speed))  # Farbskala erstellen
        legend_html = colormap._repr_html_()  # Erzeugt die HTML-Darstellung der Farbskala
        # HTML-Widget erstellen
        legend_widget = HTML(value=f"""
        <div style="position: relative; z-index:9999; background-color: white; padding: 10px;">
            <h4>Windgeschwindigkeit (m/s)</h4>
            {legend_html}
        </div>
        """)

        # WidgetControl zur Karte hinzufügen
        legend_control = WidgetControl(widget=legend_widget, position="bottomright")
        m.add_control(legend_control)

        m.add(FullScreenControl())

        return m

# Shiny App erstellen und starten
app = App(app_ui, server)



# to run: shiny run app2.py
