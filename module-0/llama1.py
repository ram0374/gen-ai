import os
import redis
import concurrent.futures
from typing import TypedDict, Dict

from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import SerpAPIWrapper, PythonREPLTool
from langgraph.graph import StateGraph

# Set API keys
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["SERPAPI_API_KEY"] = "your_serpapi_api_key"

# Initialize Redis memory
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def store_memory(user_query: str, response: str):
    """Store query-response pairs in Redis for future reference."""
    redis_client.set(user_query, response)

def retrieve_memory(user_query: str) -> str:
    """Retrieve stored response for a query from Redis."""
    return redis_client.get(user_query) or "No relevant memory found."

# Initialize AI model
llm = ChatOpenAI(model_name="gpt-4-mini", temperature=0)

# Define tools
tools = {
    "wikipedia": Tool(name="Wikipedia Search", func=WikipediaAPIWrapper().run, description="Searches Wikipedia articles"),
    "google_search": SerpAPIWrapper(),
    "calculator": PythonREPLTool()
}

# Define state for LangGraph
class AgentState(TypedDict):
    query: str  # User's input query
    tool_results: Dict[str, str]  # Outputs from different tools
    memory: str  # Retrieved memory for the query

def execute_tools_parallel(state: AgentState) -> AgentState:
    """Executes multiple tools in parallel and stores results."""
    query = state["query"]
    tool_results = {}

    def run_tool(tool_name, tool):
        """Helper function to execute a tool."""
        return tool_name, tool.run(query)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_tool, name, tool): name for name, tool in tools.items()}
        for future in concurrent.futures.as_completed(futures):
            tool_name, result = future.result()
            tool_results[tool_name] = result

    state["tool_results"] = tool_results
    return state

def retrieve_past_memory(state: AgentState) -> AgentState:
    """Retrieve past memory from Redis for context-aware responses."""
    state["memory"] = retrieve_memory(state["query"])
    return state

def generate_final_response(state: AgentState) -> str:
    """Generate AI response using memory and tool outputs."""
    prompt = f"""
    User Query: {state['query']}
    Memory: {state['memory']}
    Tool Results: {state['tool_results']}
    Provide a concise and informative response.
    """
    
    response = llm.predict(prompt)
    
    # Store conversation in memory
    store_memory(state["query"], response)
    
    return response

# Build LangGraph workflow
workflow = StateGraph(AgentState)

workflow.add_node("retrieve_memory", retrieve_past_memory)
workflow.add_node("execute_tools", execute_tools_parallel)
workflow.add_node("generate_response", generate_final_response)

workflow.set_entry_point("retrieve_memory")

# Define workflow edges
workflow.add_edge("retrieve_memory", "execute_tools")
workflow.add_edge("execute_tools", "generate_response")

# Compile LangGraph executor
agent_executor = workflow.compile()

# Sample Queries
queries = [
    {"query": "Who is the current president of the United States?", "tool_results": {}, "memory": ""},
    {"query": "What is the square root of 256?", "tool_results": {}, "memory": ""},
    {"query": "Tell me about the history of Python programming language.", "tool_results": {}, "memory": ""}
]

# Execute queries
for query in queries:
    response = agent_executor.invoke(query)
    print("\n📝 Query:", query["query"])
    print("💡 Response:", response)
