import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load .env if available (for local dev)
load_dotenv()

st.set_page_config(page_title="LangChain RAG with Groq", page_icon="🧠")
st.title("🧠 LangChain RAG App with Groq & LangChain Docs")

# Load VectorStore
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("vectorstore/faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# Load Groq LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Input
query = st.text_input("Ask a question about LangChain:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke(query)
        st.success("Answer:")
        st.write(result["result"])

        # Optional: show source chunks
        with st.expander("📄 Source Documents"):
            for i, doc in enumerate(result['source_documents']):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
