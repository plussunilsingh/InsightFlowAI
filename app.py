import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
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
    embedding = OllamaEmbeddings(
        model="nomic-embed-text"
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
        # Ensure we use similarity_search_with_score for accurate results
        results = vector_db.similarity_search_with_score(cricketer, k=3)
        
        for idx, (doc, score) in enumerate(results):
            # Calculate similarity percentage assuming L2 distance
            similarity_percent = max(round((1 - score / 2) * 100, 2), 0)
            st.write(f"- **Rank {idx + 1}** | **Similarity:** {similarity_percent}% | **Match:** {doc.page_content[:100]}...")
        st.markdown("---")

# Run only when query is entered
if query:

    st.subheader("🔍 Retrieving Relevant Chunks...")

    # Similarity search with score (returns distance, lower is better)
    results = vector_db.similarity_search_with_score(
        query,
        k=3
    )

    context_text = ""

    # Similarity threshold
    SIMILARITY_THRESHOLD = 1.0

    # Display retrieved chunks
    for idx, (doc, score) in enumerate(results):

        # Convert distance score to percentage (assuming L2 max ~2)
        similarity_percent = round((1 - score / 2) * 100, 2)

        # Prevent negative %
        similarity_percent = max(similarity_percent, 0)

        st.markdown(f"### Chunk {idx + 1}")

        # Progress bar
        st.progress(min(int(similarity_percent), 100))

        # Confidence level
        if similarity_percent > 80:
            st.success(f"📊 High Match Confidence: {similarity_percent}%")

        elif similarity_percent > 50:
            st.warning(f"📊 Medium Match Confidence: {similarity_percent}%")

        else:
            st.error(f"📊 Low Match Confidence: {similarity_percent}%")

        # Show chunk only if relevant
        if score < SIMILARITY_THRESHOLD:

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