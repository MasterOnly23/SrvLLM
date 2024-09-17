from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Usar tu token de Hugging Face si es necesario
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  
access_token = os.getenv('HUGGING_FACE_TOKEN')  # Obtener el token de Hugging Face desde las variables de entorno
cache_dir = "D:/SrvLLM/LLM_model"  # Ruta donde se guardará el modelo y tokenizer

# Verificar que el token se haya cargado correctamente
if not access_token:
    print("Error: No se pudo cargar el token de Hugging Face desde el archivo .env")
    exit(1)

# Cargar el tokenizer y el modelo
try:
    print("Descargando el tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=access_token, force_download=True)
    print("Tokenizer descargado correctamente.")
except Exception as e:
    print(f"Error al descargar el tokenizer: {e}")
    exit(1)

# Cargar el modelo
try:
    print("Descargando el modelo...")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, token=access_token)
    print("Modelo descargado correctamente.")
except Exception as e:
    print(f"Error al descargar el modelo: {e}")
    exit(1)

print("Modelo y tokenizer descargados correctamente.")
