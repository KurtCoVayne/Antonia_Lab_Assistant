import sounddevice as sd
import soundfile as sf
import numpy as np
import os

# ── Configuración ──────────────────────────────────────
SAMPLE_RATE = 44100  # Calidad estándar
CHANNELS = 1         # Mono (suficiente para voz)
DURATION = 5         # Segundos
FILENAME = "prueba_antonia.wav"
INPUT_DEVICE = 0     # Tu Micrófono USB PnP

# ── Grabación ──────────────────────────────────────────
print(f"Grabando {DURATION} segundos... ¡Habla ahora!")

# Grabamos directamente desde el hardware (dispositivo 0)
audio = sd.rec(int(DURATION * SAMPLE_RATE), 
               samplerate=SAMPLE_RATE, 
               channels=CHANNELS, 
               dtype='float32', 
               device=INPUT_DEVICE)

sd.wait()  # Esperar a que termine de grabar

# ── Diagnóstico y Guardado ──────────────────────────────
# Calculamos la amplitud máxima para saber si el mic captó algo
max_amp = np.max(np.abs(audio))
print(f"Grabación terminada.")
print(f"Propiedades: Shape={audio.shape}, Amplitud Máxima={max_amp:.4f}")

# Guardamos el archivo en el SSD
sf.write(FILENAME, audio, SAMPLE_RATE)
print(f"✅ Archivo guardado como: {os.path.abspath(FILENAME)}")

# ── Intento de reproducción (local en Jetson) ─────────
print("Reproduciendo en las salidas físicas de la Jetson...")
sd.play(audio, SAMPLE_RATE)
sd.wait()
print("Proceso terminado.")