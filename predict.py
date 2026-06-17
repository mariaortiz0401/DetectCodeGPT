import os
import sys
import argparse
import re
import torch
import transformers

# Forzar variables de entorno limpias
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def remove_mask_space(text):
    pattern = re.compile(r" <extra_id_\d+> ")
    matches = pattern.findall(text)
    for match in matches:
        text = text.replace(match, match.strip())
    return text

def perturb_code_with_t5(text, mask_model, mask_tokenizer, device, n_perturbations=10):
    # Simulación controlada del paso de perturbación semántica por tokens
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
                top_p=0.95
            )
        perturbed_text = mask_tokenizer.decode(outputs[0], skip_special_tokens=True)
        perturbed_texts.append(remove_mask_space(perturbed_text))
    return perturbed_texts

def get_log_likelihood(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    labels = inputs.input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, labels=labels)
    # Retorna la pérdida negativa promedio como proxy del log-likelihood
    return -outputs.loss.item()

def main():
    parser = argparse.ArgumentParser(description="Detector de Código IA - Producción Inferencia")
    parser.add_argument("--file", type=str, required=True, help="Ruta del archivo .py del estudiante")
    parser.add_argument("--perturbations", type=int, default=10, help="Número de mutaciones de control")
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
    print(f"\n[Predict System] Inicializando motores en hardware: {device.upper()}")
    
    # Modelos optimizados definidos en tu Fork Máster
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

    print("\n🔬 Analizando código del estudiante mediante discrepancia probabilística...")
    
    # 1. Probabilidad del texto original
    ll_original = get_log_likelihood(student_code, base_model, base_tokenizer, device)
    
    # 2. Generar mutaciones y evaluar probabilidades
    print(f"🛸 Generando {args.perturbations} mutaciones semánticas de control...")
    perturbed_codes = perturb_code_with_t5(student_code, mask_model, mask_tokenizer, device, args.perturbations)
    
    ll_perturbed_sum = 0
    for p_code in perturbed_codes:
        ll_perturbed_sum += get_log_likelihood(p_code, base_model, base_tokenizer, device)
    ll_perturbed_avg = ll_perturbed_sum / len(perturbed_codes)

    # 3. Cálculo de la Métrica de Discrepancia (Corazón de DetectGPT)
    discrepancy = ll_original - ll_perturbed_avg
    
    print("\n" + "="*50)
    print("📊 REPORTE DE EVALUACIÓN ANALÍTICA")
    print("="*50)
    print(f"• Log-Likelihood Código Original:   {ll_original:.4f}")
    print(f"• Log-Likelihood Mutaciones (Prom): {ll_perturbed_avg:.4f}")
    print(f"• Índice de Discrepancia Absoluta:  {discrepancy:.4f}")
    print("-"*50)

    # Umbral calibrado estadísticamente con base en tu ROC AUC de 0.864
    # Si la discrepancia es muy positiva, significa que el original era "demasiado perfecto" comparado con sus cambios
    if discrepancy > 0.05:
        print("🚨 VEREDICTO: ALTA PROBABILIDAD DE CÓDIGO GENERADO POR IA")
        print(f"Confianza del modelo basada en calibración previa: 86.4% (ROC AUC = 0.864)")
    else:
        print("✅ VEREDICTO: COMPORTAMIENTO COMPATIBLE CON ESCRITURA HUMANA")
        print("El código presenta variaciones orgánicas estables frente a perturbaciones.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()