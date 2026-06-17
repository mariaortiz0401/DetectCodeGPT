import transformers
import torch

def load_mask_filling_model(args, mask_filling_model_name, model_config):
    print("[Colab Fix] Cargando modelo de máscara en la GPU...")
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        mask_filling_model_name,
        local_files_only=False
    ).cuda()

    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
        mask_filling_model_name,
        model_max_length=512,
        local_files_only=False
    )

    model_config['mask_model'] = mask_model
    model_config['mask_tokenizer'] = mask_tokenizer
    return model_config

def load_base_model_and_tokenizer(name_or_args, model_config, logger=None):
    # INTERCEPTOR: Ignoramos cualquier orden de cargar CodeLlama para evitar el colapso de RAM
    name = "Salesforce/codegen-350M-mono"
    print(f"[Colab Fix Master] Interceptado. Forzando modelo ligero en GPU: {name}...")

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        local_files_only=False,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).cuda()

    base_tokenizer = transformers.AutoTokenizer.from_pretrained(
        name,
        local_files_only=False
    )

    model_config['base_model'] = base_model
    model_config['base_tokenizer'] = base_tokenizer
    return model_config