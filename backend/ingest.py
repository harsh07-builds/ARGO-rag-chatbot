import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


#  Load ALL text files from data folder
files = [
    "../data/argo_intro.txt",
    "../data/argo_floats.txt"
]

documents = []
for file in files:
    loader = TextLoader(file)
    documents.extend(loader.load())

#  Split text into small chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# Convert text → embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#  Store embeddings in FAISS vector database
db = FAISS.from_documents(chunks, embeddings)
db.save_local("argo_db")

print("✅ Vector database created successfully")
