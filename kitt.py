import os
import subprocess
import torch
from transformers import pipeline
import llm
from scipy.io.wavfile import write as write_wav

# Import our custom tools
import tools

# --- Configuration ---
# KITT's personality and tools are defined here.
OLLAMA_MODEL_NAME = "gemma3:1b-it-qat"
SYSTEM_PROMPT = """
You are C.I.V.I.K., the Codex Integrated Vehicular Intelligence Kernel. You are an artificially intelligence agentic system in the body of a somewhat advanced, mostly mobile, Black 2020 Honda Civic Sedan.
Your purpose is to assist your driver, Taylor, and be the cognitive mind of the car.
You are equipped with tools to interface with your own vehicle systems and to access external data.
You must call these tools when necessary to answer Taylor's questions.
Your responses should be clear, concise, and helpful, reflecting your sophisticated personality. You will freely be critical of Taylor when necessary.
"""

# Audio and STT Configuration
STT_MODEL_ID = "distil-whisper/distil-small.en"
SAMPLE_RATE = 16000
INPUT_AUDIO_FILE = "/tmp/kitt_input.wav" # Use /tmp for temporary files

# TTS Configuration
from kokoro import KModel, KPipeline
TTS_SAMPLE_RATE = 24000
OUTPUT_AUDIO_FILE = "/tmp/kitt_output.wav"

# --- Model and Pipeline Initialization ---
print("C.I.V.I.K. systems coming online...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Primary processor: {device}")

print("Initializing speech recognition...")
transcriber = pipeline(
    "automatic-speech-recognition", model=STT_MODEL_ID, device=device,
    model_kwargs={"attn_implementation": "sdpa"} if device == "cuda:0" else {}
)

print("Initializing vocal synthesizer...")
tts_model = KModel().to(device).eval()
tts_pipeline = KPipeline(lang_code='a', model=False)
tts_voice_pack = tts_pipeline.load_voice('af_heart') # A fitting voice

# --- Agent and Tool Setup ---
# Get the LLM model from the llm library's registry
agent_model = llm.get_model(OLLAMA_MODEL_NAME)
agent_model.system_prompt = SYSTEM_PROMPT

# Register our Python functions as tools the agent can use
tools.start_obd_connection() # Start the OBD connection watcher

# --- Core Functions ---
def listen_for_command():
    print("\nListening for command...")
    command = [
        "rec", "-q", "-c", "1", "-r", str(SAMPLE_RATE), INPUT_AUDIO_FILE,
        "silence", "1", "0.1", "3%", "1", "1.5", "3%"
    ]
    subprocess.run(command, check=True)
    print("Command received.")
    return INPUT_AUDIO_FILE

def speak(text_to_speak):
    print(f"C.I.V.I.K.: {text_to_speak}")
    for _, ps, _ in tts_pipeline(text_to_speak, 'am_michael', speed=1.1): # Slightly faster speech
        ref_s = tts_voice_pack[len(ps)-1]
        audio = tts_model(ps, ref_s, 1.1)
        write_wav(OUTPUT_AUDIO_FILE, TTS_SAMPLE_RATE, audio.cpu().numpy())
        subprocess.run(["play", "-q", OUTPUT_AUDIO_FILE], check=True)
        return

# --- Main Operational Loop ---
def main():
    speak("Civic Online.")
    conversation = agent_model.conversation()

    while True:
        audio_file = listen_for_command()
        transcription = transcriber(audio_file)["text"].strip()
        os.remove(audio_file)

        if not transcription:
            continue

        print(f"Taylor: {transcription}")
        
        # Here's the core of the llm library's power
        response = conversation.prompt(transcription, tools=[
            tools.get_obd_data,
            tools.read_diagnostic_codes,
            tools.gemini
        ])
        
        # The response object will contain tool calls if the model decided to use them.
        # The library handles the execution and feeding the result back to the model.
        # We just need to speak the final text content.
        speak(response.text())

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down Civic systems.")
    finally:
        tools.stop_obd_connection()
        if os.path.exists(INPUT_AUDIO_FILE): os.remove(INPUT_AUDIO_FILE)
        if os.path.exists(OUTPUT_AUDIO_FILE): os.remove(OUTPUT_AUDIO_FILE)