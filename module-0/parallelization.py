

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
import os
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
# Set API keys
os.environ["OPENAI_API_KEY"] = "sk-proj--fQnuUJZ7E8jT3BlbkFJXsS_-66RYdBzRtF_war-Tj0I48kpAsbFu_zPxFgucgOGERa9L7VMtXpM0oTJwHcuyD7QeNadcA"
os.environ["SERPAPI_API_KEY"] = "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"
HF_Token = "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"
class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]
    
import os, getpass
# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")
# _set_env("TAVILY_API_KEY")



def search_web(state):
    
    """ Retrieve docs from web search """

    # Search
    print(state['question'])
    tavily_search = TavilySearchResults(max_results=1)
    search_docs = tavily_search.invoke(state['question'])

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )
    print("search_web", formatted_search_docs)
    return {"context": [formatted_search_docs]} 

def search_wikipedia(state):
    
    """ Retrieve docs from wikipedia """

    print(state['question'])
    # Search
    search_docs = WikipediaLoader(query=state['question'], 
                                  load_max_docs=1).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    print("search_wikipedia", formatted_search_docs)
    return {"context": [formatted_search_docs]} 

def search_duckduckgo(state):
    
    """ Retrieve docs from duckduckgo """

    # Search
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchResults
    wrapper = DuckDuckGoSearchAPIWrapper( max_results=1)   #region="de-de", time="d", max_results=1)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")
    search_docs = search.invoke(state['question'])
     # Format
    import re
    # Regular expression to extract snippets and links
    pattern = r"snippet:\s(.*?),\s+title:.*?,\s+link:\s(https?://\S+)"

    matches = re.findall(pattern, search_docs)

    # Store results in a list of dictionaries
    #parsed_results = [{"snippet": snippet, "link": link} for snippet, link in matches]

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{link}">\n{snippet}\n</Document>'
            for snippet, link in matches
        ]
    )
    print("------")
    print("search_duckduckgo", formatted_search_docs)
    return {"context": [formatted_search_docs]}


def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return {"context": a * b}


# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return {"context": a + b}

def generate_answer(state):
    
    """ Node to answer a question """

    # Get state
    context = state["context"]
    question = state["question"]
    print("question", state['question'])
    print(" context", state['context'])

    # Template
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, 
                                                       context=context)    
    
    # Answer
    answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content=f"Answer the question.")])
      
    # Append it to state
    return {"answer": answer}
tools = [add, multiply, search_web]
# Add nodes
builder = StateGraph(State)
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
# Initialize each node with node_secret 
#builder.add_node("search_web",search_web)
builder.add_node("tools", ToolNode(tools))
#builder.add_node("multiply", multiply)
builder.add_node("generate_answer", generate_answer)

# Flow
builder.add_edge(START, "tools")
#builder.add_edge(START, "multiply")
#builder.add_edge(START, "search_web")
builder.add_edge("tools", "generate_answer")
#builder.add_edge("add", "generate_answer")
#builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()

#display(Image(graph.get_graph().draw_mermaid_png()))    
with open("output1.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())

user_input = input("Please enter something: ")

# Print the input
question ={"question": f"{user_input}"}
print(question)
print(f"You entered: {question}")
result = graph.invoke(question)
print(result['answer'].content) 
    