from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.huggingface import HuggingFaceLLM
from smolagents import OpenAIServerModel

import os

PHOENIX_API_KEY = "6e181e3690e27691a3f:48f729b"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

os.environ["HF_TOKEN"]="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"
model = OpenAIServerModel(
            model_id="qwen2.5:3b",
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )
# llm = HuggingFaceLLM(
#     model_name="meta-llama/Llama-3.2-3B-Instruct",
#     context_window=3900,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},

#     device_map="auto",
# )

# model.complete("Hello, how are you?")
messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
response = model(messages, stop_sequences=["END"])
print(response)
# I am good, how can I help you today?