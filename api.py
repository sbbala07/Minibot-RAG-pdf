# ----------------------------------------------------
# Minibot RAG API — FastAPI Backend
# ----------------------------------------------------

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""    # Force CPU for Ollama

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate


# ----------------------------------------------------
# FASTAPI APP SETUP
# ----------------------------------------------------

app = FastAPI(title = "Minibot RAG API")

# Allow requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_methods= ["*"],
    allow_headers= ["*"],
)

# Folder to save uploaded PDFs
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------------------------------
# RAG PIPELINE SETUP
# ----------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

llm = OllamaLLM(model="llama3.2:1b")
vectorstore = None

prompt_template = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.
Answer ONLY using the context provided.
If the answer is not found, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}
"""
)

# ----------------------------------------------------
# REQUEST MODEL
# ----------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    history: list = []


# ----------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "llama3.2:1b"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore
    
    # Save uploaded file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)
    
    return {
        "message": "PDF processed successfully",
        "filename": file.filename,
        "chunks_added": len(chunks)
    }



@app.post("/chat")
async def chat(request: ChatRequest):
    global vectorstore
    
    if vectorstore is None:
        return {"error": "No PDF uploaded yet. Please upload a PDF first."}
    
    # Build conversational context
    recent_history = request.history[-6:]
    history_text = ""
    for message in recent_history:
        role = message.get("role", "").upper()
        content = message.get("content", "")
        history_text += f"{role}: {content}\n\n"
    
    search_query = history_text + f"USER: {request.question}"
    
    # Retrieve relevant chunks
    docs = vectorstore.similarity_search(search_query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Extract citations
    citations = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        source = source.split('/')[-1].split(chr(92))[-1]
        page = doc.metadata.get("page", "?")
        citation = f"{source} — Page {page + 1}"
        if citation not in citations:
            citations.append(citation)
    
    # Generate answer
    prompt = prompt_template.format(
        context=context,
        question=request.question
    )
    answer = llm.invoke(prompt)
    
    return {
        "answer": answer,
        "sources": citations
    }


# ----------------------------------------------------
# RUN SERVER
# ----------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)