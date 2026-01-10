from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint


#  FastAPI 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Schema
class Question(BaseModel):
    question: str

#  Embeddings 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS
db = FAISS.load_local(
    "argo_db",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="conversational",
        temperature=0.3,
        max_new_tokens=512,
    )
)


@app.post("/chat")
def chat(req: Question):
    query = req.question.strip()

    # 1️ Greetings 
    if query.lower() in ["hi", "hello", "hey", "yo"]:
        return {"answer": "Hello! Ask me something about ARGO ocean data."}

    # 2️ Retrieve from FAISS (NO score filtering)
    docs = db.similarity_search(query, k=3)

    # 3️ If nothing retrieved → reject
    if not docs:
        return {"answer": "I can answer questions only about the ARGO data."}

    # 4️ Build context from retrieved docs
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an expert on ARGO ocean data.
Answer the question using ONLY the context below.
If the answer is not present, say so clearly.

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm.invoke(prompt)
    return {"answer": answer.content}
