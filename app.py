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

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "query" not in st.session_state:
    st.session_state.query = ""

def set_processing():
    if st.session_state.query_input.strip():
        st.session_state.is_processing = True
        st.session_state.query = st.session_state.query_input

def stop_processing():
    st.session_state.is_processing = False
    st.session_state.query = ""

col1, col2, col3 = st.columns([6, 2, 2])
with col1:
    st.text_input(
        "", 
        placeholder="e.g. Who is Sachin Tendulkar?", 
        label_visibility="collapsed",
        key="query_input",
        disabled=st.session_state.is_processing
    )
with col2:
    st.button("🔍 Search", type="primary", use_container_width=True, 
              disabled=st.session_state.is_processing, on_click=set_processing)
with col3:
    st.button("🛑 Stop Search", type="secondary", use_container_width=True, 
              disabled=not st.session_state.is_processing, on_click=stop_processing)

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

if st.session_state.is_processing and st.session_state.query:
    active_query = st.session_state.query
    
    # 2. Generating Answer Section
    st.markdown("### 🤖 Generated Answer")
    
    with st.spinner("Retrieving relevant information..."):
        # Fast Retrieval: Use similarity_search_with_score to get scores natively and avoid slow manual re-embedding
        results = vector_db.similarity_search_with_score(
            active_query,
            k=5
        )
    
    context_text = "\n\n".join([doc.page_content for doc, distance in results])
    
    if not context_text.strip():
        st.warning("❌ No relevant information found in documents.")
    else:
        final_prompt = prompt.format(context=context_text, question=active_query)
        
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        # Use a placeholder for streaming to improve perceived performance
        answer_placeholder = st.empty()
        
        try:
            full_response = ""
            for chunk in llm.stream(final_prompt):
                full_response += chunk
                answer_placeholder.markdown(full_response + "▌")
            
            answer_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Generation stopped or failed: {e}")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 3. Retrieving Relevant Chunks Section
    st.markdown("### 🔍 Retrieved Relevant Chunks")
    
    for idx, (doc, distance) in enumerate(results):
        # Calculate similarity percentage from distance
        similarity_percent = max(round((1 - (distance / 2)) * 100, 2), 0)
        
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
    
    st.markdown("---")
    if st.button("✨ Start New Search", type="primary", use_container_width=True):
        st.session_state.is_processing = False
        st.session_state.query = ""
        st.rerun()