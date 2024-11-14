from shiny import App, reactive, ui, render
from ipyleaflet import Map, Marker, Popup
from ipywidgets import HTML
from shinywidgets import render_widget, output_widget

# Erstellen der Tab-UI
app_ui = ui.page_navbar(
    ui.nav_panel("WPP database", output_widget("map"), value="map"),
    ui.nav_panel(
        "Customise WPP",
        ui.input_numeric("lat", "Latitude", value=20),
        ui.input_numeric("lon", "Longitude", value=0),
        ui.input_slider("capacity", "Capacity", 0, 20000, 1000),
        ui.output_text("output_summary"),
        value='customise_WPP'
    ),
    id="navbar_selected",
    title="Wind Production Forecast"
)

# Server-Funktion
def server(input, output, session):
    # Funktion für die Konfiguration beim Wechsel zu "Customise WPP"
    @reactive.effect
    @reactive.event(input.entire_forecast)
    def update_turbine_configuration():
        data = input.entire_forecast()
        if data:
            # Update der Eingabefelder mit neuen Werten
            ui.update_numeric("lat", value=data['lat'])
            ui.update_numeric("lon", value=data['lon'])
            ui.update_slider("capacity", value=data['capacity'])

            # Wechsel zu "Customise WPP" Tab
            ui.update_navs("navbar_selected", selected="customise_WPP")

    # Interaktive Karte mit Marker und "Entire Forecast"-Button erstellen
    @output
    @render_widget
    def map():
        m = Map(center=(50, 10), zoom=5)

        # Marker hinzufügen
        marker = Marker(location=(50, 10))
        m.add_layer(marker)

        # Popup mit Button erstellen und den Wert von `a` direkt in den HTML-String einfügen
        popup_html = HTML(
            f'<b>Here is the marker</b><br><button onclick="Shiny.setInputValue(\'entire_forecast\', {{lat: 5, lon: 10, capacity: 5000}})">Entire Forecast</button>'
        )

        popup = Popup(location=(50, 10), child=popup_html, close_button=True)
        m.add_layer(popup)

        return m

    # Zeigt die konfigurierten Werte an
    @output
    @render.text
    def output_summary():
        return f"Latitude: {input.lat()}, Longitude: {input.lon()}, Capacity: {input.capacity()} kW"

# App erstellen und starten
app = App(app_ui, server)
app.run()
