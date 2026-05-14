import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = "chroma_db"
DOCS_PATH = "docs"

documents = []

print("Loading TXT files...")

for file in os.listdir(DOCS_PATH):

    if file.endswith(".txt"):

        file_path = os.path.join(DOCS_PATH, file)

        print("Loaded:", file)

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

        documents.extend(loader.load())

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# Delete old DB
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Embedding model
embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Create vector DB
Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=CHROMA_PATH
)

print("Vector DB created successfully!")