import torch
print(torch.cuda.is_available())
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    BloomForCausalLM, BloomTokenizerFast,
)
tokenize = LlamaTokenizer.from_pretrained('Llama-2-7b-hf')
model = LlamaForCausalLM.from_pretrained('Llama-2-7b-hf')