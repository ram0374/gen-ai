from smolagents import Tool, tool
from langchain_core.vectorstores import VectorStore


@tool
def multiply(a: float, b: float) -> float:
    """ Multiply two numbers. this takes two numbers and returns their product. 
    Args:
        a: The first number.
        b: The second number.
    Returns:
        float: The product of the two numbers.    
    """
    return a * b



@tool
def square(x: float) -> float:
    """Compute the square of a number.

    Args:
        x:  The number to be squared.

    Returns:
        float: The square of the input number.
    """
    return x * x

class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )