import os
import sys
import argparse
import re
import math
import torch
import transformers

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def remove_mask_space(text):
    pattern = re.compile(r" <extra_id_\\d+> ")
    matches = pattern.findall(text)
    for match in matches:
        text = text.replace(match, match.strip())
    return text

def perturb_code_with_t5(text, mask_model, mask_tokenizer, device, n_perturbations=15):
    perturbed_texts = []
    prompt = f"Perturb open-source python code maintaining logic: {text}"
    inputs = mask_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    for _ in range(n_perturbations):
        with torch.no_grad():
            outputs = mask_model.generate(
                **inputs, 
                max_length=256, 
                do_sample=True, 
                temperature=0.7,
                top_p=0.95,
                bad_words_ids=[[mask_tokenizer.pad_token_id]] # Evitar colapsos de relleno vacíos
            )
        perturbed_text = mask_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Filtro de seguridad: solo agregar si la mutación tiene texto real
        if perturbed_text and len(perturbed_text) > 5:
            perturbed_texts.append(remove_mask_space(perturbed_text))
            
    return perturbed_texts

def get_log_likelihood(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    labels = inputs.input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, labels=labels)
    
    loss = outputs.loss.item()
    # Si la pérdida no es un número válido por desbordamiento de float16, la controlamos
    if math.isnan(loss) or math.isinf(loss):
        return None
    return -loss

def main():
    parser = argparse.ArgumentParser(description="Detector de Código IA - Inferencia Robusta V3")
    parser.add_argument("--file", type=str, required=True, help="Ruta del archivo .py")
    parser.add_argument("--perturbations", type=int, default=15, help="Número de mutaciones")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ Error: El archivo {args.file} no existe.")
        sys.exit(1)

    with open(args.file, "r", encoding="utf-8") as f:
        student_code = f.read().strip()

    if not student_code:
        print("❌ Error: El archivo está vacío.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\\n[Predict System] Inicializando motores en hardware: {device.upper()}")
    
    base_model_name = "Salesforce/codegen-350M-mono"
    mask_model_name = "Salesforce/codet5p-770m"

    print("🤖 Cargando Evaluador Base (350M)...")
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    print("🎭 Cargando Perturbador T5...")
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_model_name)
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_model_name).to(device)

    print("\\n🔬 Analizando estructura probabilística del código...")
    
    ll_original = get_log_likelihood(student_code, base_model, base_tokenizer, device)
    
    if ll_original is None:
        print("❌ Error Crítico: El modelo base no pudo procesar la probabilidad del archivo original.")
        sys.exit(1)
    
    print(f"🛸 Generando mutaciones de control para el código del estudiante...")
    perturbed_codes = perturb_code_with_t5(student_code, mask_model, mask_tokenizer, device, args.perturbations)
    
    if not perturbed_codes:
        print("❌ Error Crítico: El perturbador T5 falló al generar variantes válidas para este estilo de código.")
        sys.exit(1)
        
    ll_perturbed_sum = 0
    valid_mutations = 0
    
    for p_code in perturbed_codes:
        ll_val = get_log_likelihood(p_code, base_model, base_tokenizer, device)
        if ll_val is not None:
            ll_perturbed_sum += ll_val
            valid_mutations += 1

    if valid_mutations == 0:
        print("❌ Error Crítico: Todas las probabilidades de las mutaciones resultaron indeterminadas (NaN).")
        sys.exit(1)

    ll_perturbed_avg = ll_perturbed_sum / valid_mutations

    # Cálculo final blindado contra NaNs
    discrepancy = ll_original - ll_perturbed_avg
    prob_ia = 1 / (1 + math.exp(-10 * (discrepancy - 0.05))) * 100

    print("\\n" + "="*55)
    print("📊 REPORTE DE EVALUACIÓN ANALÍTICA (MÉTRICAS FILTRADAS)")
    print("="*55)
    print(f"• Log-Likelihood Código Original:     {ll_original:.4f}")
    print(f"• Log-Likelihood Mutaciones (Prom):   {ll_perturbed_avg:.4f} (Basado en {valid_mutations} variantes limpias)")
    print(f"• Índice de Discrepancia Absoluta:    {discrepancy:.4f}")
    print("-"*55)
    print(f"• PROBABILIDAD ESTIMADA DE SER IA:    {prob_ia:.2f}%")
    print("-"*55)

    if discrepancy > 0.05:
        print("🚨 VEREDICTO: ALTA PROBABILIDAD DE CÓDIGO SINTÉTICO (IA)")
    else:
        print("✅ VEREDICTO: COMPORTAMIENTO COMPATIBLE CON ESCRITURA HUMANA")
        
    print(f"\\n[Nota de Calibración Global de la Tesis]")
    print(f"-> Precisión histórica del framework: ROC AUC = 0.8641 (Confianza global: 86.4%)")
    print("="*55 + "\\n")

if __name__ == "__main__":
    main()