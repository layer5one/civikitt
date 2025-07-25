import obd
import subprocess
import json

# --- OBD-II Connection ---
# This establishes a persistent connection to the vehicle.
print("Initializing vehicle interface...")
connection = obd.Async()

def get_obd_data(command_name: str) -> str:
    """
    Queries the vehicle's OBD-II port for a specific piece of data.
    Available commands: SPEED, RPM, THROTTLE_POS, ENGINE_LOAD, COOLANT_TEMP
    """
    if not connection.is_connected():
        return "Error: Vehicle's onboard diagnostic system is not connected."

    command_map = {
        "SPEED": obd.commands.SPEED,
        "RPM": obd.commands.RPM,
        "THROTTLE_POS": obd.commands.THROTTLE_POS,
        "ENGINE_LOAD": obd.commands.ENGINE_LOAD,
        "COOLANT_TEMP": obd.commands.COOLANT_TEMP,
    }

    if command_name.upper() not in command_map:
        return f"Error: Unknown command '{command_name}'. Use one of: {list(command_map.keys())}"

    response = connection.query(command_map[command_name.upper()])

    if response.is_null():
        return f"Error: No data received for {command_name}."

    return f"{response.value}"

def read_diagnostic_codes() -> str:
    """
    Reads and returns any Diagnostic Trouble Codes (DTCs) from the vehicle.
    """
    if not connection.is_connected():
        return "Error: Vehicle not connected."
    
    response = connection.query(obd.commands.GET_DTC)
    
    if response.is_null() or not response.value:
        return "No diagnostic trouble codes found. All systems are operating nominally."
        
    return f"Active Diagnostic Trouble Codes: {response.value}"

# --- External Tools ---
def gemini(prompt: str) -> str:
    """
    Consults the Gemini Progenitor model via the command line for complex queries
    that require external knowledge or advanced reasoning.
    """
    try:
        # We add '-m' to ensure the output is raw markdown, not the rich GUI.
        result = subprocess.run(
            ['geminicli', prompt, '-m'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except FileNotFoundError:
        return "Error: The 'geminicli' command is not installed or not in the system's PATH."
    except Exception as e:
        return f"An error occurred while running Gemini CLI: {e}"

# Watch for OBD-II connection changes
def start_obd_connection():
    connection.watch(obd.commands.RPM) # Watch a common command
    connection.start()
    print("OBD-II connection watcher started.")

def stop_obd_connection():
    if connection.is_connected():
        connection.stop()
        connection.close()
        print("OBD-II connection stopped.")