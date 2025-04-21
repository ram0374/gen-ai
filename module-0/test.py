import torch
# from llama_index.core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from transformers import BitsAndBytesConfig
from openai import OpenAI
# from llama_index.llms.huggingface import HuggingFaceLLM 
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
client = OpenAI(
  api_key=""
)
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt


def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


# quantize to save memory
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )
tools = [add, multiply]
# llm = HuggingFaceLLM(
#     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     context_window=3000,
#     max_new_tokens=256,
#    # model_kwargs={"quantization_config": quantization_config},
#     generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
#     messages_to_prompt=messages_to_prompt,
#     completion_to_prompt=completion_to_prompt,
#     device_map="auto",
# )

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )
# chat_model = ChatHuggingFace(llm=llm)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)  
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
#response = chat_model.complete("What is the meaning of life?")
response = llm_with_tools.invoke("What is the meaning of life? in 10 words" )

# from langgraph.graph import MessagesState
# from langchain_core.messages import HumanMessage, SystemMessage

# # System message
# sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# # Node
# def assistant(state: MessagesState):
#    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
print(str(response))
