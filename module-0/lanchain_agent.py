import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
#from langchain.tools import SerpAPIWrapper, PythonREPLTool

# Set API keys
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "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"

# Initialize AI model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Define tools
wikipedia_tool = Tool(
    name="Wikipedia",
    func=WikipediaAPIWrapper().run,
    description="Useful for getting summaries of topics from Wikipedia."
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

#google_search_tool = SerpAPIWrapper()

#calculator_tool = PythonREPLTool()

# List of tools
tools = [wikipedia_tool, add, multiply]

from langchain.tools import Tool

# def my_custom_tool(input_text: str):
#     return f"Processed: {input_text}"

# tools = [
#     wikipedia_tool,
#     Tool(
#         name="multiply",
#         func=multiply,
#         description="Multiply two numbers."
#     ),
#     Tool(
#         name="add",
#         func=add,
#         description="add two numbers."
#     )
# ]


# Add conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Dynamically selects tools
    verbose=True,
    memory=memory
)

# Run queries
queries = [
    "What is the capital of France?",
    "What is 25 * 40?"
]

for query in queries:
    print("\n📝 Query:", query)
    response = agent.run(query)
    print("💡 Response:", response)
