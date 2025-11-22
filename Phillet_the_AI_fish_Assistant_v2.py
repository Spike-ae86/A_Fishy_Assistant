import os 
import openai
import sounddevice as sd
import numpy as np
import json
import queue
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv
from gtts import gTTS
import subprocess
import scipy.signal
import soundfile as sf
from scipy.signal import resample
import difflib

# Load environment variables and OpenAI API key
load_dotenv()
api_key = os.getenv('OPENAI_KEY')
openai.api_key = api_key

if not api_key:
    raise ValueError("OpenAI API key not found! Please set it in your .env file.")

# Select your mic device
DEVICE_INDEX = 1  # Your Xbox mic device index

# Auto-detect sample rate
device_info = sd.query_devices(DEVICE_INDEX)
SAMPLERATE = int(device_info['default_samplerate'])

# Set input config
sd.default.device = DEVICE_INDEX
sd.default.samplerate = SAMPLERATE
sd.default.channels = 1

print(f"[DEBUG] Using device {DEVICE_INDEX}: {device_info['name']}")
print(f"[DEBUG] Default sample rate: {SAMPLERATE} Hz")

# AI Assistant configs
name = "Master"
greetings = [
    f"What's up, {name}?",
    "Yes, Master?",
    "Hello, Master of Awesomeness. How can I assist you today?",
]

# Load Vosk model
MODEL_PATH = "vosk-model-small-en-us-0.15"  # Or upgrade to vosk-model-en-us-0.22
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Vosk model not found! Download from: https://alphacephei.com/vosk/models")

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, 16000)
audio_queue = queue.Queue()

# Software mic amplifier
def amplify(audio, factor=5.0):
    return np.clip(audio * factor, -32768, 32767).astype(np.int16)

# Optional noise filter
def reduce_noise(audio_data, samplerate):
    sos = scipy.signal.butter(10, 100, 'hp', fs=samplerate, output='sos')
    return scipy.signal.sosfilt(sos, audio_data)

# TTS with gTTS + ffplay
def speak_text(text):
    print("Speaking:", text)
    temp_dir = "D:\\Python projects\\Phillet_the_AI_fish_Assistant_v2\\temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_audio_path = os.path.join(temp_dir, "temp_audio.mp3")

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(temp_audio_path)

    ffplay_path = "C:\\pypath\\ffmpeg\\bin\\ffplay.exe"
    subprocess.run([ffplay_path, "-nodisp", "-autoexit", temp_audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.remove(temp_audio_path)

# Record from mic
def record_audio(duration=5):
    print(f"[DEBUG] Recording audio for {duration} seconds at {SAMPLERATE} Hz...")
    audio_data = sd.rec(int(duration * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype=np.int16)
    sd.wait()
    audio_data = audio_data.flatten()

    # Save for debug
    sf.write("debug.wav", audio_data, SAMPLERATE)
    return audio_data

# Fuzzy match for wake word
def match_wake_word(text):
    clean_text = text.lower()
    candidates = ["hey fishy", "a fishy", "fishy", "hey fish"]
    match = difflib.get_close_matches(clean_text, candidates, n=1, cutoff=0.6)
    if match:
        print(f"[DEBUG] Wake word matched: '{match[0]}' from input '{clean_text}'")
    return bool(match)

# Recognize speech with Vosk
def recognize_speech(audio_data):
    print("[DEBUG] Amplifying + filtering audio...")
    amplified = amplify(audio_data)
    filtered = reduce_noise(amplified, SAMPLERATE)

    print("[DEBUG] Resampling to 16kHz...")
    resampled = resample(filtered, int(len(filtered) * 16000 / SAMPLERATE)).astype(np.int16)

    print("[DEBUG] Running Vosk recognition...")
    recognizer.AcceptWaveform(resampled.tobytes())
    result = json.loads(recognizer.Result())
    print("[DEBUG] Recognition result:", result)
    return result.get("text", "")

# Wake word listener
def listen_for_wake_word():
    print("Listening for 'Hey fishy'...")
    while True:
        audio_data = record_audio(duration=7)
        text = recognize_speech(audio_data)
        print(f"[DEBUG] Recognized text: '{text}'")
        if match_wake_word(text):
            print("Wake word detected.")
            speak_text(np.random.choice(greetings))
            listen_and_respond()
            break

# GPT loop
def listen_and_respond():
    messages = []
    while True:
        print("Listening for command...")
        audio_data = record_audio(duration=10)
        text = recognize_speech(audio_data)

        if not text:
            print("No speech detected. Try again.")
            continue

        print(f"You said: {text}")
        messages.append({"role": "user", "content": text})

        response = send_to_chatGPT(messages)
        print(f"GPT: {response}")
        speak_text(response)

# ChatGPT call
def send_to_chatGPT(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.5,
    )
    message = response.choices[0].message["content"]
    messages.append({"role": "assistant", "content": message})
    return message

# Entry point
if __name__ == "__main__":
    listen_for_wake_word()
# This script listens for a wake word, then records audio commands and responds using OpenAI's GPT model.
# Make sure to have the required libraries installed and the Vosk model downloaded. 