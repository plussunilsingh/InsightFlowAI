import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import numpy as np

# Constants
CHROMA_PATH = "chroma_db"
DOCS_PATH = "docs"

# Streamlit page config
st.set_page_config(
    page_title="ContextIQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
        color: #1E88E5;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        color: #6c757d;
    }
    .answer-box {
        background-color: #f8f9fa;
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Caching & Setup -----------------

@st.cache_resource(show_spinner="Loading Models & Database...")
def load_models():
    # Load embedding model
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

    # Load vector database
    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    # Load local LLM
    llm = OllamaLLM(model="llama3.1", temperature=0.0)
    
    return embedding, vector_db, llm

embedding, vector_db, llm = load_models()

# Sidebar Setup
st.sidebar.title("🧠 ContextIQ Settings")
st.sidebar.markdown("---")

# Get stats
try:
    num_files = len([f for f in os.listdir(DOCS_PATH) if f.endswith(".txt")])
except FileNotFoundError:
    num_files = 0

try:
    num_chunks = vector_db._collection.count()
except Exception:
    num_chunks = 0

st.sidebar.subheader("📊 Knowledge Base Stats")
st.sidebar.info(f"📄 Loaded Files: {num_files}")
st.sidebar.info(f"🧩 Total Chunks: {num_chunks}")
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Active Models")
st.sidebar.success("🔗 Embedding: bge-large-en-v1.5")
st.sidebar.success("🤖 LLM: llama3.1")

if st.sidebar.button("🗑️ Clear Cache & Reload"):
    st.cache_resource.clear()
    st.rerun()

# ----------------- UI Layout -----------------

st.markdown('<p class="main-header">🧠 ContextIQ</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions from your local knowledge base</p>', unsafe_allow_html=True)

# 1. Question Input Section
st.markdown("### 📝 Ask a Question")
query = st.text_input(
    "",
    placeholder="e.g. Who is Sachin Tendulkar?",
    label_visibility="collapsed"
)

st.markdown("---")

PROMPT_TEMPLATE = """
You are an expert AI assistant designed to extract information from provided context.

Context:
{context}

Question:
{question}

Instructions:
1. Carefully read the Context provided above.
2. Answer the Question based strictly on the Context. 
3. You are allowed to rephrase the information and connect names with their descriptions.
4. If you can deduce the answer from the facts in the Context, do so.
5. If the Context absolutely does not contain the answer, reply ONLY with: "I could not find relevant information."
6. Keep your answer direct and professional. Do not add filler words.
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def get_cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

if query:
    
    # 2. Generating Answer Section
    st.markdown("### 🤖 Generated Answer")
    
    # We retrieve first, but display answer first
    # MMR retrieval
    results = vector_db.max_marginal_relevance_search(
        query,
        k=5,
        fetch_k=20
    )
    
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    if not context_text.strip():
        st.warning("❌ No relevant information found in documents.")
    else:
        final_prompt = prompt.format(context=context_text, question=query)
        
        with st.spinner("Generating answer..."):
            response = llm.invoke(final_prompt)
            
        st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)
        
    # 3. Retrieving Relevant Chunks Section
    st.markdown("### 🔍 Retrieved Relevant Chunks")
    
    # Embed query to compute scores manually for MMR results
    query_emb = embedding.embed_query(query)
    
    for idx, doc in enumerate(results):
        # We manually compute similarity score since MMR doesn't return scores directly
        doc_emb = embedding.embed_query(doc.page_content)
        similarity = get_cosine_similarity(query_emb, doc_emb)
        similarity_percent = max(round(similarity * 100, 2), 0)
        
        source = doc.metadata.get("source", "Unknown")
        
        # Color coding logic
        if similarity_percent > 75:
            color = "🟢" # High
            bar_color = "success"
        elif similarity_percent > 55:
            color = "🟡" # Medium
            bar_color = "warning"
        else:
            color = "🔴" # Low
            bar_color = "error"
            
        with st.expander(f"{color} Chunk {idx + 1} | Source: {source} | Confidence: {similarity_percent}%"):
            
            if bar_color == "success":
                st.success(f"High Match Confidence: {similarity_percent}%")
            elif bar_color == "warning":
                st.warning(f"Medium Match Confidence: {similarity_percent}%")
            else:
                st.error(f"Low Match Confidence: {similarity_percent}%")
                
            st.info(doc.page_content)