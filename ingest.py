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
