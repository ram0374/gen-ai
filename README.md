# gen-ai
This project showcases an Agentic RAG architecture that leverages autonomous agents to reason, retrieve, and generate answers over structured and unstructured knowledge sources. Built using HuggingFace’s SmolAgent, LangChain, and LangGraph, the system enables intelligent task decomposition...

HuggingFace SmolAgent: Lightweight and composable agent framework that handles task planning, tool usage, and execution. Utilized to build autonomous agents capable of reasoning over knowledge graphs and retrieved documents.

LangChain: Provides the backbone for LLM orchestration — chaining together tools, memory, retrievers, and prompts. Used to:

Interface with vector stores (e.g., FAISS, Chroma).

Wrap tools like web search, code interpreters, or document loaders.

Handle input parsing and output formatting.

LangGraph: Introduces graph-based control flow for complex workflows involving agents. Used to:

Manage conversation state and branching logic.

Implement multi-step agent loops with memory and conditional transitions.

Handle retry, reflection, or sub-agent delegation strategies.



