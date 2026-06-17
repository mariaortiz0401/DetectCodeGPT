import os
import json
import random
import argparse
import transformers
import torch
from tqdm import tqdm

# =====================================================================
# CORRECCIÓN DE LA TESIS: Función defensiva para cargar cualquier JSONL
# =====================================================================
def load_data(path, max_num=1000):
    """
    Carga y parsea de forma robusta los archivos del dataset (como The Vault o locales),
    garantizando que no se rompa el pipeline si faltan columnas específicas.
    """
    print(f"[Fork Fix] Cargando datos desde la ruta unificada: {path}")
    prompts = []
    solutions = []

    if not os.path.exists(path):
        print(f"⚠️ Alerta: El archivo no existe en la ruta especificada: {path}")
        return prompts, solutions

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                
                # PARCHE CRUCIAL 1: Extracción tolerante de la solución de código
                solution = data.get("code", data.get("solution", data.get("output", "")))
                
                # PARCHE CRUCIAL 2: Extracción tolerante del prompt/docstring
                prompt = data.get("docstring", data.get("prompt", data.get("instruction", "")))

                # PARCHE CRUCIAL 3: Si venía de CodeSearchNet estructurado mapeamos tokens, 
                # si viene de The Vault o tus archivos .py lo emulamos en caliente
                if "docstring_tokens" not in data and prompt:
                    data["docstring_tokens"] = prompt.split()

                if solution and prompt:
                    solutions.append(solution)
                    prompts.append(prompt)
                    
            except Exception as e:
                # Si una línea está corrupta o incompleta, la salta en lugar de colapsar
                continue

            if len(prompts) >= max_num:
                break

    print(f"✅ Carga completada. Se prepararon {len(prompts)} muestras válidas para el modelo.")
    return prompts, solutions

# =====================================================================
# PIPELINE COMPLETO DEL AUTOR ENCASTRADO CON TUS PARCHES
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline de Generación de Código Saneado")
    parser.add_argument('--dataset', type=str, default='CodeSearchNet')
    parser.add_argument('--dataset_key', type=str, default='train')
    parser.add_argument('--model_name', type=str, default='Salesforce/codegen-350M-mono')
    parser.add_argument('--max_num', type=int, default=100)
    parser.add_argument('--output_name', type=str, default='outputs')
    args = parser.parse_args()

    # Construcción dinámica de rutas del repositorio
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else "."
    train_path = os.path.join(base_dir, "data", "CodeSearchNet", "python", f"{args.dataset_key}.jsonl")

    # Inyección de tu lógica limpia de carga
    prompts, solutions = load_data(path=train_path, max_num=args.max_num)

    if len(prompts) == 0:
        print("❌ Error crítico: No se pudieron procesar muestras de entrenamiento.")
        return

    # Entorno de ejecución adaptativo para Colab / Mac M5
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Fork Fix] Inicializando inferencia en dispositivo: {device.upper()}")

    print(f"🤖 Cargando modelo generativo base: {args.model_name}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name, local_files_only=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        local_files_only=False,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # Creamos la carpeta destino de los artefactos generados si no existe
    output_dir = os.path.join(base_dir, "code-generation", "output", args.dataset, args.output_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "generated_codes.txt")

    print(f"⚡ Ejecutando inferencia sobre las muestras extraídas...")
    with open(output_file, "w", encoding="utf-8") as f_out:
        for idx in tqdm(range(len(prompts)), desc="Generando código"):
            prompt_text = prompts[idx]
            
            # Tokenizar e inferir de forma segura
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128, num_return_sequences=1, do_sample=True, top_p=0.95)
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Guardamos el resultado estructurado respetando el diseño del framework
            meta_record = {
                "id": idx,
                "prompt": prompt_text,
                "human_solution": solutions[idx],
                "output": generated_text
            }
            f_out.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

    print(f"🎉 ¡Proceso completado con éxito!")
    print(f"📁 Los códigos e históricos se han guardado en: {os.path.abspath(output_file)}")

if __name__ == '__main__':
    main()