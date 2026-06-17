import os
import json
import logging
import argparse
import torch
import transformers
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_data(input_path, output_path, max_num=100):
    print(f"[Fork Inferencia] Leyendo prompts desde: {input_path}")
    print(f"[Fork Inferencia] Escribiendo dataset de contraste en: {output_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"❌ El archivo de entrada no existe: {input_path}")
        return

    # 1. Configurar hardware y cargar el LLM ligero
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model_name = "Salesforce/codegen-350M-mono"
    
    print(f"🤖 Cargando generador ligero en [{device.upper()}]: {model_name}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0

    # 2. Procesar el archivo de streaming línea por línea
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        # Envolvemos el iterador en tqdm para ver la barra de progreso de generación
        for line in tqdm(f_in, desc="Generando código con IA", total=max_num):
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                
                # Extraer datos reales de The Vault
                human_code = data.get("code") or data.get("solution") or data.get("output", "")
                prompt_text = data.get("docstring") or data.get("prompt") or data.get("instruction", "")
                
                if not human_code.strip() or not prompt_text.strip():
                    continue 
                
                # 3. Inferencia: Forzar a la IA a escribir su propia solución basada en el prompt
                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=256).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_length=128, 
                        num_return_sequences=1, 
                        do_sample=True, 
                        top_p=0.95,
                        temperature=0.2 # Temperatura baja para código más estructurado y predecible
                    )
                
                ai_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 4. EL CONTRATO DE LA TESIS: Rompemos el espejo
                structured_line = {
                    "solution": human_code,  # Código Real de GitHub (The Vault)
                    "output": ai_code        # Código Artificial Creado por la IA
                }
                
                f_out.write(json.dumps(structured_line, ensure_ascii=False) + '\n')
                count += 1
                
            except Exception as e:
                continue

            if count >= max_num:
                break

    logger.info(f"🎉 ¡Éxito! Dataset de contraste creado: {count} muestras guardadas en {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_num', type=int, default=50) # 50 o 100 es ideal para pruebas rápidas
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    repo_root = os.path.dirname(script_dir)
    
    input_file = os.path.join(repo_root, 'data', 'CodeSearchNet', 'python', 'train.jsonl')
    output_file = os.path.join(repo_root, 'code-generation', 'output', 'TheVault', 'CodeLlama-7b-hf-10000-tp0.2', 'outputs.txt')

    generate_synthetic_data(input_file, output_file, max_num=args.max_num)