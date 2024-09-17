from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os
# Paso 3: Cargar el modelo y tokenizer desde la ubicación local
def cargar_modelo_local(cache_dir):
    try:
        print("Intentando cargar el modelo desde la ubicación local...")
        model = AutoModelForCausalLM.from_pretrained(cache_dir)
        print("Modelo cargado correctamente.")
        tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        print("Tokenizer cargado correctamente.")
        return model, tokenizer
    except Exception as e:
        print(f"Error al cargar el modelo o tokenizer: {e}")
        return None, None

# Paso 4: Cargar el dataset
def cargar_dataset(dataset_path):
    try:
        print("Cargando dataset...")
        dataset = load_dataset('csv', data_files=dataset_path)
        print("Dataset cargado correctamente.")
        return dataset
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None

# Paso 5: Configurar y entrenar el modelo
def entrenar_modelo(model, tokenizer, dataset, output_dir):
    print("Configurando el entrenamiento...")

    training_args = TrainingArguments(
        output_dir=output_dir,               # Carpeta de salida
        per_device_train_batch_size=4,       # Tamaño de batch por dispositivo
        per_device_eval_batch_size=4,        # Tamaño de batch para evaluación
        evaluation_strategy="steps",         # Evaluar cada ciertos pasos
        logging_steps=10,                    # Pasos para registrar logs
        save_steps=500,                      # Pasos para guardar el modelo
        num_train_epochs=3,                  # Número de épocas
        weight_decay=0.01,                   # Decaimiento de peso
        push_to_hub=False,                   # No subir a Hub por ahora
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
    )

    print("Iniciando el entrenamiento...")
    trainer.train()
    print("Entrenamiento completado.")

# Función principal
def main():
    # cache_dir = "D:/SrvLLM/LLM_model"  # Ruta donde se encuentra el modelo localmente
    cache_dir = "D:/SrvLLM/LLM_model/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f/"
    output_dir = "./results"  # Ruta donde se guardará el modelo entrenado
    dataset_path = "./customer_assistant_data.csv"  # Ruta del dataset

    # Cargar modelo y tokenizer
    model, tokenizer = cargar_modelo_local(cache_dir)
    if model is None or tokenizer is None:
        print("No se pudo cargar el modelo y tokenizer. Terminando ejecución.")
        return

    # Cargar dataset
    dataset = cargar_dataset(dataset_path)
    if dataset is None:
        print("No se pudo cargar el dataset. Terminando ejecución.")
        return

    # Entrenar el modelo
    entrenar_modelo(model, tokenizer, dataset, output_dir)

if __name__ == "__main__":
    main()