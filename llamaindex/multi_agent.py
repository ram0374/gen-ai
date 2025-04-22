from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)
from llama_index.core import VectorStoreIndex
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from utility import get_llm
import asyncio
import chromadb
from llama_index.core.workflow import Context

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b



# Create a vector store
db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create a query engine
embed_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
llm = get_llm()
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=embed_model
)
query_engine = index.as_query_engine(llm=llm)
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="personas",
    description="descriptions for various types of personas",
    return_direct=False,
)
# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
llm = get_llm()
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="personas",
    description="descriptions for various types of personas",
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. ",
    tools=[query_engine_tool],
    llm=llm
)

# Create and run the workflow
agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)

async def main(msg: str):
    response = await agent.run(msg)
    print(response)
    
msg = "Can you add 5 and 3?"

# Ensure only one event loop is used
try:
    loop = asyncio.get_running_loop()
except RuntimeError:  
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

loop.run_until_complete(main(msg))

msg = "Search the database for  fiction and return some persona descriptions."
loop.run_until_complete(main(msg))

