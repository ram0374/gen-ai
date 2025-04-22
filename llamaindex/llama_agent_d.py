import os
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

# initialize llm


# initialize agent
# agent = AgentWorkflow.from_tools_or_functions(
#     [FunctionTool.from_defaults(multiply)],
#     llm=llm
# )

PHOENIX_API_KEY = "6e181e3690e27691a3f:48f729b"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

os.environ["HF_TOKEN"]="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"
# model = OpenAIServerModel(
#             model_id="qwen2.5:1.5b-instruct",
#             api_base="http://localhost:11434/v1",
#             api_key="ollama"
#         )


import transformers

# Define the custom model class
class CustomPipeline(transformers.Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # This method is called when the pipeline needs to generate text based on an input.
    def generate(self, context):
        # Your specific logic here. For instance:
        # return self.model.generate(context)
        pass
    
# Instantiate your custom model
model_id = "qwen2.5:1.5b-instruct"
api_base = "http://localhost:11434/v1"
api_key = "ollama"

model_path = f"~/.ollama/models/{model_id}" # Replace with the actual path
pipe = transformers.pipeline("text-generation", model=model_path) # Or any other task as needed

# custom_pipeline_model = transformers.pipeline(
#     task="text-generation",
#     model=model_id,
#     api_token=api_key,
# )

# Your main function to handle inputs and outputs
def process_input(input_text):
    generated_output = custom_pipeline_model.generate(input_text)
    return generated_output

# Example usage:
input_text = "What is the weather like in Beijing?"
output = process_input(input_text)
print(output)
    