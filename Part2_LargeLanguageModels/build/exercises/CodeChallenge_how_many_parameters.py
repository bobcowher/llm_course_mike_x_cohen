from transformers import GPT2Tokenizer, GPT2Model

dict_keys = (["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
# dict_keys = (["gpt2", "gpt2-medium"])

models = {}
tokenizers = {}

for key in dict_keys:
    models[key] = GPT2Model.from_pretrained(key)
    tokenizers[key] = GPT2Tokenizer.from_pretrained(key)

for name, model in models.items():

    param_count = 0

    for param in model.parameters():
        param_count += param.numel()
    
    print(f"Name: {name} Trainable Parameters: {param_count:,}")

   


