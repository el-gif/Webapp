from shiny import App, ui
from shinywidgets import render_widget, output_widget
from ipyleaflet import Map, Marker
from ipywidgets import HTML

app_ui = ui.page_fluid(
    output_widget("map")
)

def server(input, output, session):
    @output
    @render_widget
    def map():
        # Zentriere die Karte auf die Koordinaten
        m = Map(center=(52.204793, 360.121558), zoom=9)

        # Erstelle den Marker und deaktiviere das Verschieben
        marker = Marker(
            location=(52.1, 359.9),
            draggable=False,  # Marker kann nicht verschoben werden
            title="hello"  # Tooltip für den Marker
        )

        # Erstelle den Popup-Inhalt
        popup_content = HTML(value=f"<strong>Wind speed forecast:</strong> select forecast step<br>"
                                   f"<strong>Production forecast:</strong> select forecast step<br>")

        # Popup dem Marker zuweisen
        marker.popup = popup_content

        # Füge den Marker der Karte hinzu
        m.add(marker)

        return m

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
