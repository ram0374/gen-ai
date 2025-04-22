from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.huggingface import HuggingFaceLLM

import os

PHOENIX_API_KEY = "6e181e3690e27691a3f:48f729b"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

os.environ["HF_TOKEN"]="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"
# OpenAIServerModel(
#             model_id=model_id,
#             api_base="http://localhost:11434/v1",
#             api_key="ollama"
#         )
llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},

    device_map="auto",
)

llm.complete("Hello, how are you?")
# I am good, how can I help you today?