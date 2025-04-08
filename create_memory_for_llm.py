from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Step 1: Load raw PDF(s)
import os
from langchain_community.document_loaders import PDFPlumberLoader

# Path to your data folder
DATA_PATH = "data/"

def load_pdf_files(data_path):
    """Load all PDFs from the specified directory with error handling."""
    documents = []
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data_path, file)
            print(f"Loading: {pdf_path}")
            try:
                loader = PDFPlumberLoader(pdf_path)
                doc = loader.load()
                documents.extend(doc)
                print(f"✅ Successfully loaded {len(doc)} pages from {file}")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
    return documents

# Load PDFs
documents = load_pdf_files(DATA_PATH)
#print("📚 Total loaded PDF pages:", len(documents))

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
#print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings 

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)