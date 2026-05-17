import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Constants
CHROMA_PATH = "chroma_db"

# Streamlit page config
st.set_page_config(
    page_title="ContextIQ",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 ContextIQ - Local RAG Assistant")

st.markdown("---")

# Optimize model load by caching models at app start
@st.cache_resource
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
    llm = OllamaLLM(model="llama3.1")
    
    return embedding, vector_db, llm

embedding, vector_db, llm = load_models()

# Prompt template
PROMPT_TEMPLATE = """
You are an enterprise knowledge assistant.

Answer ONLY from the provided context.

If the answer is not found in the context,
say:
"I could not find relevant information."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# User input
query = st.text_input(
    "Ask a question from your documents:"
)

st.sidebar.title("🛠️ Testing Menu")
run_tests = st.sidebar.button("Run Keyword & Cricketer Tests")

if run_tests:
    st.subheader("🧪 Testing All Cricketer Names and Keywords")
    cricketers = [
        "Anil Kumble", "Jasprit Bumrah", "Kapil Dev", "MS Dhoni",
        "Rahul Dravid", "Rohit Sharma", "Sachin Tendulkar", "Sourav Ganguly",
        "Virat Kohli", "Yuvraj Singh"
    ]
    
    for cricketer in cricketers:
        st.markdown(f"**Query:** `{cricketer}`")
        # Ensure we use MMR search for accurate and diverse results
        results = vector_db.max_marginal_relevance_search(cricketer, k=5, fetch_k=20)
        
        for idx, doc in enumerate(results):
            st.write(f"- **Rank {idx + 1}** | **Source:** {doc.metadata.get('source', 'Unknown')} | **Match:** {doc.page_content[:100]}...")
        st.markdown("---")

# Run only when query is entered
if query:

    st.subheader("🔍 Retrieving Relevant Chunks...")

    # MMR search
    results = vector_db.max_marginal_relevance_search(
        query,
        k=5,
        fetch_k=20
    )

    context_text = ""

    # Display retrieved chunks
    for idx, doc in enumerate(results):
        st.markdown(f"### Chunk {idx + 1} (Source: {doc.metadata.get('source', 'Unknown')})")
        st.info(doc.page_content)
        context_text += doc.page_content + "\n\n"

    st.markdown("---")

    # No relevant information
    if context_text == "":

        st.warning("❌ No relevant information found in documents.")

    else:

        st.subheader("🤖 Generating Answer...")

        # Create final prompt
        final_prompt = prompt.format(
            context=context_text,
            question=query
        )

        # Generate response
        response = llm.invoke(final_prompt)

        # Display response
        st.success(response)