def tokenizer_lookup(model_id):
    tokenizers = {
        "mistral:7b-instruct-v0.3-q4_0": "mistralai/Mistral-7B-Instruct-v0.3",
        "llama3.1:8b-instruct-q4_K_M": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3.2:3b-instruct-q4_K_M": "meta-llama/Meta-Llama-3.2-3B-Instruct",
    }

    return tokenizers[model_id]
