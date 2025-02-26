from shiny import App, ui, reactive, render

# Example choices
selectable_turbine_types = ["Vestas V90", "Siemens SWT-2.3", "Enercon E-82", "unknown", "custom"]
known_turbine_types = ["Vestas V90", "Siemens SWT-2.3", "Enercon E-82"]

app_ui = ui.page_fluid(
    ui.input_select(
        "turbine_type", "Turbine Type",
        choices=selectable_turbine_types, selected=known_turbine_types[0]
    ),
    ui.input_text("custom_turbine", "Enter Custom Turbine Type", value="", placeholder="Type here..."),
    ui.output_text("selected_turbine")
)

def server(input, output, session):
    @reactive.calc
    def selected_turbine():
        if input.turbine_type() == "custom":
            return input.custom_turbine()
        else:
            return input.turbine_type()

    @output
    @render.text
    def selected_turbine():
        return f"Selected Turbine: {selected_turbine()}"

    # Enable/Disable the text field dynamically
    @reactive.effect
    def _():
        if input.turbine_type() == "custom":
            ui.enable("custom_turbine")
        else:
            ui.disable("custom_turbine")
            ui.update_text("custom_turbine", value="")  # Clear input when disabled

app = App(app_ui, server)
app.run()
