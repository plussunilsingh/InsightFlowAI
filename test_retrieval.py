from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embedding, collection_metadata={"hnsw:space": "cosine"})

query = "who is sachin?"
results = vector_db.similarity_search_with_score(query, k=5)

for doc, distance in results:
    similarity_percent = max(round((1 - distance) * 100, 2), 0)
    print(f"{doc.metadata['source']}: distance={distance}, sim={similarity_percent}%")
