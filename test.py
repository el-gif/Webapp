# import datetime
# import asyncio
# from shiny import App, ui, reactive, render

# # Function to calculate the next scheduled time (12:25 UTC for 13:25 UTC+1)
# def next_scheduled_time():
#     now = datetime.datetime.now(datetime.timezone.utc)
#     scheduled_time = now.replace(hour=12, minute=25, second=0, microsecond=0)

#     # If it's already past today's scheduled time, schedule for the next day
#     if now >= scheduled_time:
#         scheduled_time += datetime.timedelta(days=1)

#     return scheduled_time

# # Function to print "Hello" at the scheduled time
# async def schedule_hello():
#     while True:
#         next_time = next_scheduled_time()
#         wait_seconds = (next_time - datetime.datetime.now(datetime.timezone.utc)).total_seconds()

#         print(f"Next 'Hello' scheduled for: {next_time} UTC")

#         await asyncio.sleep(wait_seconds)  # Wait until 12:25 UTC
#         print("Hello")  # Print "Hello" to the console
#         last_hello_time.set(f"Last 'Hello' at: {next_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

# # UI layout
# app_ui = ui.page_fluid(
#     ui.h2("Hello Scheduler"),
#     ui.p("This web app prints 'Hello' every day at 13:25 UTC+1 (12:25 UTC)."),
#     ui.output_text("hello_time")
# )

# # Server function
# def server(input, output, session):
#     global last_hello_time
#     last_hello_time = reactive.Value("Waiting for the first 'Hello'...")

#     # Start the background task inside the Shiny session
#     asyncio.create_task(schedule_hello())

#     # Display the last scheduled "Hello" time
#     @output
#     @render.text
#     def hello_time():
#         return last_hello_time.get()

# # Run the Shiny app
# app = App(app_ui, server)

# # Start the app normally (no need for asyncio.run())
# if __name__ == "__main__":
#     app.run()

import datetime
from shiny import App, ui, reactive, render

# UI
app_ui = ui.page_fluid(
    ui.h2("Session Tracking"),
    ui.p("This app logs when a new session starts and when it ends."),
    ui.output_text("session_info")
)

# Server function
def server(input, output, session):
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔵 New session started at {start_time} (Session ID: {id(session)})")

    # Store session start time in a reactive value
    session_info = reactive.Value(f"Session started at: {start_time}")

    # Detect when the session ends
    def on_session_end():
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🔴 Session {id(session)} ended at {end_time}")

    session.on_ended(on_session_end)

    # Display session start time in UI
    @output
    @render.text
    def session_text():
        return session_info.get()

# Run the Shiny app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
