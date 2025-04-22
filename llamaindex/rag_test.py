from datasets import load_dataset
from pathlib import Path
from llama_index.core import SimpleDirectoryReader

import torch
torch.mps.empty_cache()
       
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from transformers import AutoTokenizer
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_MEMORY_GROWTH"] = "1"
#from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
# Login using e.g. `huggingface-cli login` to access this dataset
#dataset = load_dataset("manycore-research/SpatialLM-Testset", split="test")
dataset = load_dataset(path="dvilasuero/finepersonas-v0.1-tiny", split="train")
print(len(dataset))
Path("data").mkdir(parents=True, exist_ok=True)
#for i, p in [(i, p) for i, p in enumerate(dataset) ]:
for i, p in enumerate(dataset):
    if i > 1000:
        break
    with open(Path("data") / f"Spatial_{i}.txt", "w") as f:
        f.write(p["persona"])


reader = SimpleDirectoryReader(input_dir="data")
documents = reader.load_data()
print(len(documents)) 


db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection(name="alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

#nodes =  pipeline.arun(documents=documents[:10])


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, embed_model=embed_model
)
print(index)

from llama_index.llms.huggingface import HuggingFaceLLM

#import nest_asyncio
# Load Meta-LLaMA Tokenizer
llm_name = "meta-llama/Llama-3.2-3B-Instruct"
#tokenizer = AutoTokenizer.from_pretrained(llm_name)

#nest_asyncio.apply()  # This is needed to run the query engine
# llm = HuggingFaceLLM(model_name="meta-llama/Llama-3.2-3B-Instruct" ,  device_map="mps")



# Load LLaMA Model with Correct Tokenizer
llm = HuggingFaceLLM(model_name=llm_name, device_map="mps")
#llm = HuggingFaceLLM(model_name="StabilityAI/stablelm-tuned-alpha-3b" ,  device_map="cpu")

#llm = HuggingFaceLLM(model_name="Qwen/Qwen2.5-Coder-7B-Instruct")
# model = llm.to(torch.float16).to("mps")
# model = model.to("cpu")
# Create service context with LLM and embedding model
query_engine = index.as_query_engine(
    llm=llm,
    #response_mode="tree_summarize",
    choice_batch_size=1,
)


response = query_engine.query(
    "Respond using a persona that describes author and travel experiences?"
)
print(response)