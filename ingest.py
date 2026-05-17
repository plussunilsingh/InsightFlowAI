import os
import shutil
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma_db"
DOCS_PATH = "docs"

documents = []

print("Loading TXT files...")

for file in os.listdir(DOCS_PATH):
    if file.endswith(".txt"):
        with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
            text = f.read()

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file}
            )
        )
        print("Loaded:", file)

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# Delete old DB
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

# Create vector DB
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=CHROMA_PATH
)

print("Vector DB created successfully!")