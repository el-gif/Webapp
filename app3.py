from shiny import App, ui
from ipywidgets import SelectionSlider, Layout, Play, VBox, jslink, Dropdown, HTML, Checkbox
from shinywidgets import render_widget, output_widget
from ipyleaflet import Map, Heatmap, WidgetControl, ColormapControl, LayersControl, LayerGroup, basemaps, basemap_to_tiles, FullScreenControl, ImageOverlay, TileLayer
from ipyleaflet.velocity import Velocity
import numpy as np
import itertools
from branca.colormap import linear
import xarray as xr
from PIL import Image
import io
from base64 import b64encode
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from ipyevents import Event

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

colormap = linear.viridis.scale(round(min_wind_speed), round(max_wind_speed))

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

    # Funktion, die bei Sitzungsende ausgeführt wird
    def session_ended():
        print("Session beendet. Benutzer hat die Verbindung getrennt.")
        global initial
        initial = 0

    # Registriere die Funktion für das Sitzungsende
    session.on_ended(session_ended)

    # Render das Leaflet-Widget mit render_widget, already called at initial connection
    @output
    @render_widget
    def map():

        # Basemap-Dropdown-Optionen, https://ipyleaflet.readthedocs.io/en/latest/map_and_basemaps/basemaps.html
        basemaps_dict = {
            "OpenStreetMap": basemaps.OpenStreetMap.Mapnik,
            "OpenTopoMap": basemaps.OpenTopoMap,
            "NASAGIBS ViirsEarthAtNight": basemaps.NASAGIBS.ViirsEarthAtNight2012,
            "Google Map": TileLayer(
                url='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                attribution='Google',
                base=True  # Als Baselayer markieren
            ),
        }

        # Initiale Basemap setzen
        m = Map(center=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
                zoom=5,
                layout=Layout(width='100%', height='100vh'),
                value='Google Maps'
            )

        # Dropdown-Menü zur Auswahl der Basemap
        dropdown = Dropdown(
            options=list(basemaps_dict.keys()),  # Basemap-Namen als Optionen
            description='Basemap'
        )

        dropdown_control = WidgetControl(widget=dropdown, position='topright')
        m.add(dropdown_control)

        # Aktiviere die erste Basemap
        current_basemap_layer = basemap_to_tiles(basemaps_dict[dropdown.value])
        m.add(current_basemap_layer)

        # Funktion zur Aktualisierung der Basemap
        def update_basemap(change):
            nonlocal current_basemap_layer
            # Entferne die aktuelle Basemap
            m.remove_layer(current_basemap_layer)
            # Füge die neu ausgewählte Basemap hinzu
            current_basemap_layer = basemap_to_tiles(basemaps_dict[dropdown.value])
            m.add(current_basemap_layer)

        # Verbinde das Dropdown-Menü mit der Update-Funktion
        dropdown.observe(update_basemap, names='value')

        # Erstelle einen Slider als WidgetControl und füge ihn zur Karte hinzu
        play = Play(min=0, max=total_hours, step=step_size_hours, value=0, interval = 500, description='Time Step') # 500 ms pro Schritt
        slider = SelectionSlider(options=valid_times_interpol, value=valid_times_interpol[0], description='Time')
        jslink((play, 'value'), (slider, 'index'))
        slider_box = VBox([play, slider])

        slider_control = WidgetControl(widget=slider_box, position='topright')
        m.add(slider_control)

        # Funktion zur Aktualisierung der Karte basierend auf dem Slider
        def update_map(change):

            visibilities = []
            for i, layer in enumerate(m.layers):
                if isinstance(layer, LayerGroup):
                    visibilities[i].append({'name':layer.name, 'visibility':layer.visible})
                    m.remove(layer)
            
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

            height = len(total_selection_interpol[step_index])
            width = len(total_selection_interpol[step_index][0])

            rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    hex_color = colormap(total_selection_interpol[step_index][i][j])  # Get hex color from colormap

                    # Convert hex color to RGBA
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    a = 255  # Full opacity

                    rgba_image[i, j] = (r, g, b, a)

            # Convert the RGBA array to an image
            img = Image.fromarray(rgba_image, 'RGBA')          

            # Annahme: img ist dein bereits generiertes Bild aus dem RGBA-Array
            img_array = np.array(img)

            # Definiere die Parameter
            src_crs = 'EPSG:4326'
            dst_crs = 'EPSG:3857'

            # Berechne die Transformationsparameter für EPSG:4326 -> EPSG:3857
            src_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
            dst_transform, width, height = calculate_default_transform(src_crs, dst_crs, width, height, lon_min, lat_min, lon_max, lat_max)

            # Ziel-Array für die Reprojektion erstellen
            destination = np.zeros((4, height, width), dtype=np.uint8)

            # Reprojektion des Bildes durchführen
            reproject(
                source=img_array.transpose(2, 0, 1),
                destination=destination,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )

            im = Image.fromarray(destination.transpose(1, 2, 0), 'RGBA')

            # Konvertiere das transformierte Bild in Base64
            buffer = io.BytesIO()
            im.save(buffer, format='PNG')
            buffer.seek(0)
            img_base64 = b64encode(buffer.read()).decode('utf-8')
            img_data_url = f"data:image/png;base64,{img_base64}"

            # Erstelle das ImageOverlay mit den transformierten Daten
            image_overlay = ImageOverlay(
                url=img_data_url,
                bounds=[(lat_min, lon_min), (lat_max, lon_max)], # use bounds in degree, even if map has another CRS
                opacity=0.6
            )
            
            heatmap_layer = LayerGroup(layers=(heatmap,), name="Heatmap Layer")
            #m.add(heatmap_layer)

            velocity_layer = LayerGroup(layers=(velocity,), name="Velocity Layer")
            #m.add(velocity_layer)

            image_layer = LayerGroup(layers=(image_overlay,), name="Colormap Layer")
            #m.add(image_layer)

            layers = [heatmap_layer, velocity_layer, image_layer]

            for i in range(len(visibilities)):
                if visibilities[i]['visibility']:
                    name = visibilities[i]['name']
                    for layer in layers:
                        if name == layer.name:
                            m.add(layer)

            # Erstelle eine Checkbox für jeden Layer zur Kontrolle der Sichtbarkeit
            if initial == 0:
                for layer in layers:
                    m.add(layer)

                heatmap_checkbox = Checkbox(value=True, description="Heatmap Layer")
                velocity_checkbox = Checkbox(value=True, description="Velocity Layer")
                colormap_checkbox = Checkbox(value=True, description="Colormap Layer")

                # Platziere die Checkboxes auf der Karte
                layer_control_box = VBox([heatmap_checkbox, velocity_checkbox, colormap_checkbox])
                layer_control_widget = WidgetControl(widget=layer_control_box, position='topright')

                # Füge den custom LayerControl zur Karte hinzu
                m.add_control(layer_control_widget)

            # Erstelle eine Funktion, um die Sichtbarkeit der Layer zu steuern
            def toggle_layer_visibility(change, layer):
                if change['new']:
                    layer.visible = True  # Layer sichtbar machen
                    m.add(layer)  # Layer hinzufügen
                else:
                    layer.visible = False  # Layer unsichtbar machen
                    m.remove(layer)  # Layer entfernen

            # Verknüpfe die Checkboxes mit den entsprechenden Layern
            heatmap_checkbox.observe(lambda change: toggle_layer_visibility(change, heatmap_layer), names='value')
            velocity_checkbox.observe(lambda change: toggle_layer_visibility(change, velocity_layer), names='value')
            colormap_checkbox.observe(lambda change: toggle_layer_visibility(change, image_layer), names='value')


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
        # m.add(colormap_control)

        

        
        # layer_control = LayersControl(position='bottomleft')
        # m.add(layer_control)

        # Erstellen einer benutzerdefinierten Legende als HTML
        # colormap = linear.viridis.scale(round(min_wind_speed), round(max_wind_speed))  # Farbskala erstellen
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
        m.add(legend_control)

        m.add(FullScreenControl())

        return m

# Shiny App erstellen und starten
app = App(app_ui, server)

app.run()

# to run: shiny run app3.py
