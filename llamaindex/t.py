import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Correct model ID from Hugging Face
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Configure quantization (4-bit mode)
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True, 
#     bnb_4bit_compute_dtype=torch.float32,
#     bnb_4bit_use_double_quant=True
# )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load model with proper quantization settings
model = AutoModelForCausalLM.from_pretrained(
    model_id,
   # quantization_config=quant_config,
    low_cpu_mem_usage=True,
    #torch_dtype=torch.float16,
    device_map="mps",  # Automatically map to CPU/MPS
    trust_remote_code=True,
)

# Create text-generation pipeline
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="mps",
)

# Run a sample query
prompt = "Explain the significance of artificial intelligence in modern society."
response = text_gen_pipeline(prompt, max_length=200, do_sample=True, temperature=0.7)
print(response[0]["generated_text"])
