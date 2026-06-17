import os
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(path, max_num=1000):
    print(f"Cargando datos desde: {path}")
    prompts = []
    solutions = []

    if not os.path.exists(path):
        logger.error(f"El archivo no existe: {path}")
        return prompts, solutions

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Extraemos el código y el docstring de forma segura
                solution = data.get("code", "") or data.get("solution", "")
                prompt = data.get("docstring", "") or data.get("prompt", "")

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
    parser.add_argument('--dataset_key', type=str, default='train') # 'train', 'test' o 'valid'
    args = parser.parse_args()

    # 1. ENCONTRAR LA RAÍZ REAL DEL REPOSITORIO DE FORMA DINÁMICA
    # os.path.abspath(__file__) nos da la ruta de 'generate.py'. 
    # El primer dirname nos lleva a 'code-generation', y el segundo a 'DetectCodeGPT'.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # 2. CONSTRUIR LA RUTA ABSOLUTA AL ARCHIVO DATASET
    train_path = os.path.join(repo_root, 'data', 'CodeSearchNet', 'python', f'{args.dataset_key}.jsonl')

    # Ejecutar la carga con la ruta blindada
    prompts, solutions = load_data(path=train_path, max_num=args.max_num)

    if len(prompts) == 0:
        print("❌ Error crítico: No se pudieron procesar muestras de entrenamiento.")
    else:
        print(f"¡Pipeline de datos listo! Muestras preparadas para el modelo: {len(prompts)}")