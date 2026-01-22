import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# --- Page Config ---
st.set_page_config(page_title="PragyanAI", layout="wide")
st.title("🔍 PragyanAI - RAG Chatbot")

# --- Secrets / API Key ---
# Make sure to add GROQ_API_KEY in Streamlit Cloud Settings > Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")

# --- Initialize Models (Cached to save memory) ---
@st.cache_resource
def load_models():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.2)
    return embedding_model, llm

if not groq_api_key:
    st.error("Please add your GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

embedding_model, llm = load_models()
CHROMA_DIR = "chroma_db"
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)

# --- Sidebar: File Upload ---
with st.sidebar:
    st.header("Step 1: Upload")
    uploaded_files = st.file_uploader("Upload PDF, TXT, or CSV", type=["pdf", "txt", "csv"], accept_multiple_files=True)
    index_btn = st.button("📥 Index Files")

if index_btn and uploaded_files:
    with st.spinner("Processing documents..."):
        docs = []
        for uploaded_file in uploaded_files:
            # Streamlit files are in-memory; we must save them to temp files for LangChain loaders
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                path = tmp_file.name
            
            ext = uploaded_file.name.split(".")[-1].lower()
            if ext == "pdf": loader = PyPDFLoader(path)
            elif ext == "txt": loader = TextLoader(path)
            elif ext == "csv": loader = CSVLoader(path)
            
            docs.extend(loader.load())
            os.unlink(path) # Clean up temp file

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)
        st.success(f"Indexed {len(chunks)} chunks!")

# --- Main Chat Interface ---
st.header("Step 2: Ask a Question")
query = st.text_input("What would you like to know?")

if query:
    if vectorstore._collection.count() == 0:
        st.warning("Please upload and index documents first.")
    else:
        with st.spinner("Searching and generating..."):
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            context_docs = retriever.invoke(query)
            context_text = "\n\n".join(d.page_content for d in context_docs)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer using ONLY the provided context. If unknown, say so."),
                ("human", "Context:\n{context}\n\nQuestion:\n{question}")
            ])
            
            chain = prompt | llm
            response = chain.invoke({"context": context_text, "question": query})
            st.markdown("### Response:")
            st.write(response.content)
