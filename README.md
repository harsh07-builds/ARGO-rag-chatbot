# 🌊 ARGO RAG Chatbot  
*A Retrieval-Augmented Generation (RAG) based AI system for answering questions from ARGO oceanographic data.*

---

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows a chatbot to answer user questions using **custom ARGO ocean data** instead of relying only on a general LLM.

It combines:

- 📄 Domain document ingestion  
- 🧠 Semantic embeddings  
- 🗃 Vector database (FAISS)  
- 🤖 Large Language Model reasoning  
- 🌐 Web-based chat interface  
- ⚡ FastAPI backend  

The model retrieves the most relevant document chunks and generates **grounded, hallucination-free answers**.

---

## 🧱 System Architecture

| Layer | Technology | Purpose |
|------|------------|---------|
| Frontend | HTML, TailwindCSS, JavaScript | Chat interface |
| Backend | FastAPI, Python | API & orchestration |
| Embeddings | HuggingFace Sentence Transformers | Semantic vector creation |
| Vector DB | FAISS | Fast similarity search |
| LLM | HuggingFace Inference API | Natural language generation |
| Framework | LangChain | RAG pipeline orchestration |

---

## 🔄 RAG Workflow

### 1️⃣ Data Ingestion
Text Files → Chunking → Embeddings → FAISS Index


### 2️⃣ Query Processing
User Query → Query Embedding → Similarity Search → Top-K Chunks

### 3️⃣ Answer Generation
Retrieved Context + Prompt → LLM → Final Grounded Answer


---

## ⚙️ Core Components

### 📂 Document Store
- `argo_intro.txt`
- `argo_floats.txt`
- Converted into semantic vectors

### 🧠 Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Converts text into high-dimensional meaning vectors

### 🗃 Vector Database
- FAISS (Facebook AI Similarity Search)
- Enables millisecond-level retrieval

### 🤖 LLM Reasoning
- Context-aware answer generation
- No hallucinations (strict RAG grounding)

---

## 🛠 Backend API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Accepts user query and returns RAG-based answer |

### Example Request
{
  "question": "What is the ARGO program?"
}
###Example Response
{
  "answer": "The ARGO program is a global network of autonomous floats that measure temperature and salinity in the world's oceans..."
}

🌐 Frontend Features

💬 Chat UI

🎙 Voice input (Speech-to-Text)

🔊 Audio responses (Text-to-Speech)

🌍 3D interactive globe visualization

⚡ Real-time backend communication

🚀 Setup & Run
1. Clone Repository
git clone https://github.com/harsh07-builds/ARGO-rag-chatbot.git
cd ARGO-rag-chatbot

2. Create Environment
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt

3. Build Vector Database
cd backend
python ingest.py

4.## 🔐 API Key Setup (HuggingFace)

This project uses HuggingFace Inference API for LLM responses.

### Step 1: Create Token
Go to: https://huggingface.co/settings/tokens  
Create a **Read** access token.

### Step 2: Set Environment Variable

#### Windows (PowerShell)
setx HUGGINGFACEHUB_API_TOKEN "your_token_here"


5. Start Backend
uvicorn app:app --reload

6. Open Frontend

Open frontend/index.html in browser.

🧪 Concepts Implemented
Concept	Description
Embeddings	Transform text into semantic vectors
Vector Search	Nearest-neighbor similarity retrieval
RAG	Retrieval-Augmented Generation
Grounding	LLM answers constrained to data
API Design	FastAPI endpoints
Full-Stack AI	Frontend + Backend + LLM + DB

🎯 Learning Outcomes
Built production-style RAG pipeline
Implemented vector databases
Integrated LLM inference APIs
Designed AI-driven backend services
Developed full-stack AI application
Understood hallucination control in LLMs
