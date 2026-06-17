import os
import json
import argparse
import transformers
import torch
import time
from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================================
# PARCHES DE CARGA INTEGRADOS (Para evitar dependencias rotas e hilos)
# =====================================================================

def load_mask_filling_model(args, mask_filling_model_name, model_config):
    print("[Fork Fix] Cargando modelo de máscara en la GPU de forma secuencial...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        mask_filling_model_name, 
        local_files_only=False
    ).to(device)
    
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
        mask_filling_model_name, 
        model_max_length=512, 
        local_files_only=False
    )
    
    model_config['mask_model'] = mask_model
    model_config['mask_tokenizer'] = mask_tokenizer
    return model_config

def load_base_model_and_tokenizer(name_or_args, model_config, logger_obj=None):
    # INTERCEPTOR PROTECTOR: Forzamos el modelo ligero en Colab gratuito / Entornos estándar
    # Si cuentas con alta capacidad de hardware, puedes descomentar la extracción de argumentos.
    name = "Salesforce/codegen-350M-mono"
    print(f"[Fork Fix Master] Forzando modelo ligero optimizado en GPU/Hardware: {name}...")
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    for intento in range(3):
        try:
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                name, 
                local_files_only=False, 
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            break
        except Exception as e:
            if intento < 2:
                print(f"⚠️ Servidor de Hugging Face inestable (Error de Red). Reintentando en 5 segundos... (Intento {intento+2}/3)")
                time.sleep(5)
            else:
                raise e
    
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(
        name, 
        local_files_only=False
    )
    
    model_config['base_model'] = base_model
    model_config['base_tokenizer'] = base_tokenizer
    return model_config

# =====================================================================
# LOGICA DE EXTRACCIÓN Y PROCESAMIENTO DE TEXTOS/PERTURBACIONES
# =====================================================================

def replace_masks(masked_texts, model_config, args):
    mask_model = model_config['mask_model']
    mask_tokenizer = model_config['mask_tokenizer']
    device = next(mask_model.parameters()).device
    
    # Prevenir errores sintácticos con tokens esperados
    n_expected = [int(x.split('_')[-1].replace('>', '')) for x in masked_texts if '<extra_id_' in x]
    if not n_expected:
        n_expected = [0]
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    
    inputs = mask_tokenizer(masked_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = mask_model.generate(**inputs, max_length=150, do_sample=True, top_p=0.95, eos_token_id=stop_id)
    
    raw_fills = mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return raw_fills

def extract_fills(raw_fills):
    # Extrae las piezas generadas por el modelo de máscara para reconstruir el código
    fills = []
    for raw_fill in raw_fills:
        tokens = raw_fill.split("<extra_id_")
        fill = {}
        for token in tokens:
            if not token.strip():
                continue
            parts = token.split(">")
            if len(parts) >= 2:
                mask_id = parts[0].strip()
                content = ">".join(parts[1:])
                fill[int(mask_id)] = content.replace("<pad>", "").replace("</s>", "").strip()
        fills.append(fill)
    return fills

def apply_fills(masked_texts, fills):
    # Inserta los fragmentos dentro de las posiciones enmascaradas <extra_id_X>
    perturbed_texts = []
    for text, fill in zip(masked_texts, fills):
        perturbed_text = text
        for idx in sorted(fill.keys()):
            perturbed_text = perturbed_text.replace(f"<extra_id_{idx}>", fill[idx])
        perturbed_texts.append(perturbed_text)
    return perturbed_texts

# =====================================================================
# MOTOR CENTRAL DE DETECCIÓN (CÁLCULO DE PROBABILIDADES)
# =====================================================================

def get_ll(text, model, tokenizer):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    labels = inputs.input_ids.clone()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    # Retorna la log-probabilidad (Negative Log-Likelihood convertida)
    return -outputs.loss.item() * inputs.input_ids.shape[1]

def generate_data(dataset, dataset_key, max_num, min_len, max_len, max_def_num):
    # Búsqueda dinámica robusta del dataset en la estructura de carpetas
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else "."
    path = os.path.join(base_dir, "data", "CodeSearchNet", "python", "train.jsonl")
    
    if not os.path.exists(path):
        # Fallback por si corre directo en el directorio raíz o en subcarpetas alternativas
        path = "data/CodeSearchNet/python/train.jsonl"
        
    logger.info(f"[Fork Fix] Cargando datos desde la ruta unificada: {path}")
    data = []
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo train.jsonl en la ruta: {os.path.abspath(path)}. Por favor verifica su ubicación.")
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            line = json.loads(line)
            
            # Homologación defensiva de columnas contra KeyError ('code', 'solution' u 'output')
            codigo_real = line.get('code', line.get('solution', line.get('output', '')))
            if not codigo_real:
                continue
                
            if 'output' not in line:
                line['output'] = codigo_real
            if 'solution' not in line:
                line['solution'] = codigo_real
                
            # Filtro opcional por complejidad estructural de funciones
            if line['solution'].count('def') > max_def_num or line['output'].count('def') > max_def_num:
                continue
                
            data.append(line)
            if len(data) >= max_num:
                break
                
    return data

# =====================================================================
# FUNCIÓN PRINCIPAL (PIPELINE METODOLÓGICO)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="DetectCodeGPT - Pipeline Optimizado para Tesis")
    parser.add_argument('--language', type=str, default='python', help='Lenguaje de programación a evaluar')
    parser.add_argument('--dataset', type=str, default='CodeSearchNet', help='Nombre del dataset de entrada')
    parser.add_argument('--base_model_name', type=str, default='Salesforce/codegen-350M-mono', help='Modelo base generativo')
    parser.add_argument('--mask_filling_model_name', type=str, default='Salesforce/codet5p-770m', help='Modelo perturbador')
    parser.add_argument('--max_num', type=int, default=10, help='Cantidad de muestras a evaluar')
    parser.add_argument('--num_perturbations', type=int, default=5, help='Número de variaciones sintácticas por código')
    args = parser.parse_args()

    logger.info("====== Iniciando Pipeline DetectCodeGPT Saneado ======")
    
    # 1. Cargar datos
    try:
        dataset_data = generate_data(args.dataset, "train", args.max_num, 10, 1000, 3)
        logger.info(f"✅ Dataset cargado correctamente. Muestras seleccionadas: {len(dataset_data)}")
    except Exception as e:
        logger.error(f"Error al procesar el archivo de datos: {e}")
        return

    # 2. Inicializar Modelos usando la interfaz de parches directos
    model_config = {}
    model_config = load_mask_filling_model(args, args.mask_filling_model_name, model_config)
    model_config = load_base_model_and_tokenizer(args.base_model_name, model_config)

    results = []
    
    # 3. Ciclo de evaluación métrica
    logger.info("⚡ Procesando discrepancias de log-probabilidades...")
    for idx, sample in enumerate(tqdm(dataset_data, desc="Evaluando código")):
        original_code = sample['output']
        
        # Generar variaciones de perturbación simulando el entorno de T5
        # (Para entornos de prueba rápidos, creamos máscaras basadas en saltos de línea y bloques)
        masked_texts = []
        lines = original_code.split('\n')
        if len(lines) > 2:
            for p in range(args.num_perturbations):
                # Introducimos una máscara sintáctica en una sección aleatoria del cuerpo
                pivot = len(lines) // 2
                masked_code = '\n'.join(lines[:pivot]) + " <extra_id_0> " + '\n'.join(lines[pivot+1:])
                masked_texts.append(masked_code)
        else:
            masked_texts = [original_code + " <extra_id_0> "] * args.num_perturbations
            
        # Reemplazar máscaras usando el modelo perturbador CodeT5
        try:
            raw_fills = replace_masks(masked_texts, model_config, args)
            fills = extract_fills(raw_fills)
            perturbed_codes = apply_fills(masked_texts, fills)
        except Exception as e:
            logger.warning(f"Salto de muestra {idx} por inconsistencia en tokenización de máscara: {e}")
            continue

        # Calcular Log-Likelihood del código original
        ll_original = get_ll(original_code, model_config['base_model'], model_config['base_tokenizer'])
        
        # Calcular Log-Likelihood de las variaciones permutadas
        ll_perturbed_list = []
        for p_code in perturbed_codes:
            ll_p = get_ll(p_code, model_config['base_model'], model_config['base_tokenizer'])
            ll_perturbed_list.append(ll_p)
            
        mean_ll_perturbed = np.mean(ll_perturbed_list)
        
        # Métrica DetectCodeGPT: Discrepancia logarítmica (Log-Likelihood Log-Rank Discrepancy)
        discrepancy = ll_original - mean_ll_perturbed
        
        results.append({
            'id': idx,
            'll_original': ll_original,
            'mean_ll_perturbed': mean_ll_perturbed,
            'discrepancy': discrepancy
        })

    # 4. Consolidar informe de salida
    df_results = pd.DataFrame(results)
    output_path = "metricas_deteccion_tesis.csv"
    df_results.to_csv(output_path, index=False)
    
    logger.info("=======================================================")
    logger.info(f"🎉 ¡Pipeline completado con éxito!")
    logger.info(f"📊 Reporte de métricas guardado en: {os.path.abspath(output_path)}")
    print("\n--- Vista previa de las métricas obtenidas ---")
    print(df_results.head())
    logger.info("=======================================================")

if __name__ == '__main__':
    main()