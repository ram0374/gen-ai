from langchain.agents import load_tools
from smolagents import CodeAgent, HfApiModel, Tool
import os
os.environ["SERPAPI_API_KEY"] ="275bc2883e4005b3f65481c51fb57947e59c5d17f04dace328a87b05509b7246"
search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])
model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(tools=[search_tool], model=model)

agent.run("Search for luxury entertainment ideas for a superhero-themed event, such as live performances and interactive experiences.")