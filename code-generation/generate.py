import os
import json
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(path, max_num=1000):
    print(f"[Fork] Cargando datos desde: {path}")
    prompts = []
    solutions = []

    if not os.path.exists(path):
        logger.error(f"El archivo no existe: {path}")
        return prompts, solutions

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                solution = data.get("code") or data.get("solution") or data.get("output", "")
                prompt = data.get("docstring") or data.get("prompt") or data.get("instruction", "")

                if solution and prompt:
                    solutions.append(solution)
                    prompts.append(prompt)
            except Exception:
                continue

            if len(prompts) >= max_num:
                break

    logger.info(f"Se cargaron exitosamente {len(prompts)} registros.")
    return prompts, solutions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_num', type=int, default=200)
    args = parser.parse_args()

    # 1. Encontrar rutas absolutas dinámicas para no perderse en Colab
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    repo_root = os.path.dirname(script_dir)
    
    train_path = os.path.join(repo_root, 'data', 'CodeSearchNet', 'python', 'train.jsonl')

    # 2. Cargar las muestras a memoria
    prompts, solutions = load_data(path=train_path, max_num=args.max_num)

    if len(prompts) == 0:
        print("❌ Alerta: No se pudieron extraer muestras válidas.")
    else:
        print(f"¡Pipeline de datos listo! Muestras preparadas: {len(prompts)}")
        
        # =====================================================================
        # AQUÍ ESTÁ LO QUE FALTA: Crear y volcar el archivo que busca main.py
        # =====================================================================
        archivo_salida_autor = {
            "original": solutions,         # Códigos humanos
            "sampled": solutions,          # Copia espejo simulación base
            "prompts": prompts             # Docstrings
        }
        
        # Construimos la ruta exacta que el detector buscará de forma rígida
        output_dir = os.path.join(repo_root, "code-generation", "output", "TheVault", "CodeLlama-7b-hf-10000-tp0.2")
        os.makedirs(output_dir, exist_ok=True)
        ruta_final_txt = os.path.join(output_dir, "outputs.txt")
        
        # Escribimos físicamente el archivo en el disco
        with open(ruta_final_txt, "w", encoding="utf-8") as f_out:
            f_out.write(json.dumps(archivo_salida_autor, ensure_ascii=False))
            
        print(f"📁 [CREADO] Archivo de simulación guardado con éxito en: {ruta_final_txt}")