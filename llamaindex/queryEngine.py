import chromadb

from llama_index.core import VectorStoreIndex

from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore
from utility import get_llm
from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult, AgentStream
import asyncio
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

# Create a RAG agent
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=[query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. ",
)

async def main(msg: str):
    response = await query_engine_agent.run(msg)
    # async for ev in response.stream_events():
    #     if isinstance(ev, ToolCallResult):
    #         print(" ---->")
    #         print("Called tool: ", ev.tool_name, ev.tool_kwargs, "=>", ev.tool_output)
    #     elif isinstance(ev, AgentStream):  # showing the thought process
    #         print(ev.delta, end="", flush=True)
    print(response)
    
msg = "Search the database for 'science fiction' and return some persona descriptions."

# Ensure only one event loop is used
try:
    loop = asyncio.get_running_loop()
except RuntimeError:  # No running event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

loop.run_until_complete(main(msg))

