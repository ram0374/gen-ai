from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (
    # DocxReader,
    # HWPReader,
    # PDFReader,
    # EpubReader,
    # FlatReader,
    # HTMLTagReader,
    # ImageCaptionReader,
    # ImageReader,
    # ImageVisionLLMReader,
    # IPYNBReader,
    # MarkdownReader,
    # MboxReader,
    # PptxReader,
    # PandasCSVReader,
    # VideoAudioReader,
    # UnstructuredReader,
    # PyMuPDFReader,
    # ImageTabularChartReader,
    # PagedCSVReader,
    # CSVReader,
    # RTFReader,
    XMLReader,
)

# XML Reader example
parser = XMLReader()
file_extractor = {"CCD.xml": parser}
documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()


text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Split docs and keep only unique ones
print("Splitting documents...", documents)

print(documents)
# docs_processed = text_splitter.split_documents(documents)
# print(docs_processed)
# docs_processed = []
# unique_texts = {}
# for doc in tqdm(source_docs):
#     new_docs = text_splitter.split_documents([doc])
#     for new_doc in new_docs:
#         if new_doc.page_content not in unique_texts:
#             unique_texts[new_doc.page_content] = True
#             docs_processed.append(new_doc)

print("Embedding documents... This should take a few minutes (5 minutes on MacBook with M1 Pro)")
# embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
# vectordb = FAISS.from_documents(
#     documents=docs_processed,
#     embedding=embedding_model,
#     distance_strategy=DistanceStrategy.COSINE,
# )

