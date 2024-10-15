from shiny import App, ui
from ipywidgets import Layout
from shinywidgets import render_widget, output_widget, register_widget
from ipyleaflet import Map, ColormapControl
from branca.colormap import linear

app_ui = ui.page_fluid(
    output_widget("map")  # Referenziere das Map-Widget in der UI
)

# Serverlogik für die Shiny App
def server(input, output, session):

    # Render das Leaflet-Widget mit render_widget
    @output
    @render_widget
    def map():
        # Erstelle die Karte
        m = Map(center=(51.0, 10.0), zoom=5, layout=Layout(width='100%', height='100vh'))

        # Farbskala erstellen
        colormap = linear.OrRd_06.scale(0, 30)

        # ColormapControl hinzufügen
        colormap_control = ColormapControl(
            caption="Windgeschwindigkeit (m/s)",
            colormap=colormap,
            value_min=0,
            value_max=30,
            position="bottomright"
        )

        # ColormapControl zur Karte hinzufügen
        m.add_control(colormap_control)

        return m

# Shiny App erstellen
app = App(app_ui, server)
