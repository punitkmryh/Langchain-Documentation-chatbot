# Langchain-Documentation-chatbot

Below is a **complete working LangChain RAG project** that:

✅ Uses **LangChain documentation** from `https://python.langchain.com/docs/introduction/` as context  
✅ Uses **FAISS** for local vector storage  
✅ Embeds using `sentence-transformers`  
✅ Uses **Groq LLM** (`llama3-70b-8192`)  
✅ Has a **Streamlit frontend**  
✅ Includes **GitHub structure** + **Streamlit Cloud deployment steps**

---

## 📁 Project Folder Structure

```
langchain-rag-groq/
│
├── app.py                      # Streamlit frontend
├── ingest.py                   # One-time document loader
├── requirements.txt            # All dependencies
├── .env                        # Optional for local testing
├── README.md                   # Documentation (optional)
├── vectorstore/                # Vector DB created after running ingest.py
│   └── faiss_index/            # Saved index
```

---

## 🧠 1. `ingest.py` – Load LangChain Docs into FAISS

```python
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Step 1: Load web content
loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
docs = loader.load()

# Step 2: Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Step 3: Convert to embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store in FAISS
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("vectorstore/faiss_index")

print("✅ FAISS vector store created from LangChain docs!")
```

---

## 🧠 2. `app.py` – Streamlit App with Groq & Vector Store

```python
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
```

---

## 📦 3. `requirements.txt`

```txt
streamlit
python-dotenv
langchain
langchain-community
langchainhub
langchain-groq
faiss-cpu
sentence-transformers
```

---

## 🗝️ 4. `.env` (Optional for local use)

```env
GROQ_API_KEY=your_real_groq_api_key
```

> For **Streamlit Cloud**, this will be added as a **secret**.

---

## 🚀 Deployment on Streamlit Cloud

### Step-by-Step:

1. **Push project to GitHub**  
   Repo example: `https://github.com/yourusername/langchain-rag-groq`

2. **Run `ingest.py` locally**  
   This creates `vectorstore/faiss_index/` folder. Commit this folder too.

3. **Go to Streamlit Cloud** → [https://streamlit.io/cloud](https://streamlit.io/cloud)

4. **Create a new app:**
   - Repo: your GitHub repo
   - File: `app.py`

5. **Add Secrets:**
   Go to **App → Settings → Secrets** and paste:

   ```toml
   GROQ_API_KEY = "your-real-groq-api-key"
   ```

6. **Deploy!** 🎉

---
