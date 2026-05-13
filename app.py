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
# llm = Ollama(model="llama3")
llm = OllamaLLM(model="llama3.1")

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

if query:

    st.subheader("🔍 Retrieving Relevant Chunks...")

    # Similarity search
    results = vector_db.similarity_search(
        query,
        k=1
    )

    context_text = ""

    # Display retrieved chunks
    for idx, doc in enumerate(results):

        st.markdown(f"### Chunk {idx + 1}")

        st.info(doc.page_content)

        context_text += doc.page_content + "\n\n"

    st.markdown("---")

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