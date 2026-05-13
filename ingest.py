import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Folder paths
DOCS_PATH = "docs"
CHROMA_PATH = "chroma_db"

# Load embedding model
embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Store all documents
all_docs = []

print("Loading PDFs...")

# Read all PDFs
for file in os.listdir(DOCS_PATH):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(DOCS_PATH, file)

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        all_docs.extend(docs)

        print(f"Loaded: {file}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(all_docs)

print(f"Total chunks created: {len(chunks)}")

# Create Chroma vector DB
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=CHROMA_PATH
)

print("Embeddings stored successfully in ChromaDB!")