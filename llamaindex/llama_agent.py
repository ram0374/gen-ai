from llama_index.core.agent.workflow import AgentWorkflow, ToolCallResult, AgentStream
from llama_index.core.tools import FunctionTool
import torch
import asyncio

# remembering state
from llama_index.core.workflow import Context
from utility import get_llm
llm = get_llm()

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


def divide(a: int, b: int) -> int:
    """Divide two numbers"""
    return a / b



# initialize agent
agent = AgentWorkflow.from_tools_or_functions(
    #[FunctionTool.from_defaults(multiply)],
    tools_or_functions=[subtract, multiply, divide, add],
    llm=llm,
    system_prompt="You are a math agent that can add, subtract, multiply, and divide numbers using provided tools.",
)
ctx = Context(agent)
# Run in an async event loop
async def main(msg: str):
    response = await agent.run(msg, ctx=ctx)
    async for ev in response.stream_events():
        if isinstance(ev, ToolCallResult):
            print(" ---->")
            print("Called tool: ", ev.tool_name, ev.tool_kwargs, "=>", ev.tool_output)
        elif isinstance(ev, AgentStream):  # showing the thought process
            print(ev.delta, end="", flush=True)
    print(response)
msg = "what is 2 multiple 5?"

# Ensure only one event loop is used
try:
    loop = asyncio.get_running_loop()
except RuntimeError:  # No running event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Run the async function inside the existing event loop
loop.run_until_complete(main(msg))

#msg = "multiply the previous response  with 10"
msg = "what is 2 + 2  * 5?"
loop.run_until_complete(main(msg))
