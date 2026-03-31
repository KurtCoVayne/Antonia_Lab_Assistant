from faster_whisper import WhisperModel
import os

# Definir la ruta del SSD
models_path = "/media/antonia_ssd/antonia/antonia_project/models/whisper"

# Asegurarse de que la carpeta exista
os.makedirs(models_path, exist_ok=True)

print("Iniciando descarga de modelos en el SSD...")

# Descargar Small
print("--- Bajando Small (Objetivo Primario) ---")
WhisperModel("small", device="cuda", compute_type="int8", download_root=models_path)

# Descargar Base
print("--- Bajando Base (Respaldo) ---")
WhisperModel("base", device="cuda", compute_type="int8", download_root=models_path)

print("\n¡Todo listo! Modelos guardados en:", models_path)