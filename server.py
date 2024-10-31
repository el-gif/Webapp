# server.py
from shiny import App, render, ui

app_ui = ui.page_fluid(
    ui.output_text_verbatim("server_message")
)

def server(input, output, session):
    # Sende eine Nachricht vom Server an den Client
    session.send_custom_message("simple_message", "Hello from server!")
    
    @output
    @render.text
    def server_message():
        return "Check console for the message from server."

app = App(app_ui, server)
app.run()