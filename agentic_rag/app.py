


from smolagents import HfApiModel, ToolCallingAgent, CodeAgent
from ingestion import vectordb
from tools.final_answer import FinalAnswerTool
from tools.retrival_tool import RetrieverTool, multiply, square
import os
os.environ["HF_TOKEN"]="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"

#model = HfApiModel("meta-llama/Llama-3.1-70B-Instruct")
model = HfApiModel("meta-llama/Llama-3.2-3B-Instruct")
retriever_tool = RetrieverTool(vectordb)
final_answer = FinalAnswerTool()
agent = CodeAgent(tools=[final_answer, retriever_tool, multiply, square], model=model)
agent_output = agent.run(" what is 5 multiple 3")

print(agent_output)