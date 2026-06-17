import os
import json
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_convert_data(input_path, output_path, max_num=200):
    print(f"[Fork Fix] Leyendo datos desde: {input_path}")
    print(f"[Fork Fix] Escribiendo formato nativo en: {output_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"❌ El archivo de entrada no existe: {input_path}")
        return

    # Crear los directorios de salida si no existen
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                
                # Extraemos el código de forma tolerante (The Vault usa 'code')
                raw_code = data.get("code") or data.get("solution") or data.get("output", "")
                
                if not raw_code.strip():
                    continue  # Si no hay código, nos lo saltamos
                
                # CONTRATO ESTRICTO: Creamos la línea exactamente como la exige el main.py original
                structured_line = {
                    "solution": raw_code,  # <--- Evita el KeyError: 'solution'
                    "output": raw_code     # <--- Evita el KeyError: 'output'
                }
                
                # Escribimos como una línea JSONL independiente
                f_out.write(json.dumps(structured_line, ensure_ascii=False) + '\n')
                count += 1
                
            except Exception:
                continue

            if count >= max_num:
                break

    logger.info(f"🎉 ¡Éxito! Se generaron {count} registros perfectamente estructurados en formato JSONL.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_num', type=int, default=100)
    args = parser.parse_args()

    # Rutas absolutas para no perderse en Colab
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    repo_root = os.path.dirname(script_dir)
    
    # Origen: El archivo que creaste con tu script de streaming
    input_file = os.path.join(repo_root, 'data', 'CodeSearchNet', 'python', 'train.jsonl')
    
    # Destino: La ruta exacta y fija (hardcoded) donde el main.py original irá a buscar los datos
    output_file = os.path.join(repo_root, 'code-generation', 'output', 'TheVault', 'CodeLlama-7b-hf-10000-tp0.2', 'outputs.txt')

    # Ejecutar la conversión física en disco
    load_and_convert_data(input_file, output_file, max_num=args.max_num)