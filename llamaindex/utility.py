
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# If the tokenizer doesn't have a chat template, define one
if not tokenizer.chat_template:
    tokenizer.chat_template = "{system_message}\n{user_message}\n{assistant_message}"

# initialize llm
def get_llm():
    return HuggingFaceLLM(
        model_name=model_id,
        context_window=3900,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        tokenizer=tokenizer,
        device_map="auto",
    )
