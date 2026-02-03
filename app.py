# ----------------------------------------------------
# RAG PDF Chatbot using Ollama + LangChain + FAISS + Gradio
# ----------------------------------------------------

import gradio as gr

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------
# 1. LOAD PDF DOCUMENT
# ----------------------------------------------------

print("Loading PDF...")

pdf_path = "policy.pdf"  # PDF must be in same folder
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ----------------------------------------------------
# 2. SPLIT DOCUMENT INTO SMALL CHUNKS
# ----------------------------------------------------

print("Splitting text...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# ----------------------------------------------------
# 3. CREATE EMBEDDINGS + FAISS VECTOR STORE
# ----------------------------------------------------

print("Creating embeddings & vector store...")

# Embedding model (lightweight & CPU-friendly)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Store document embeddings in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# ----------------------------------------------------
# 4. LOAD LOCAL LLM (SMALL MODEL – LOW RAM FRIENDLY)
# ----------------------------------------------------

print("Loading LLM...")

# Small LLM to avoid GPU/RAM issues
llm = OllamaLLM(model="llama3.2:1b")

# ----------------------------------------------------
# 5. PROMPT TEMPLATE FOR RAG
# ----------------------------------------------------

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
# 6. RAG CHAT FUNCTION
# ----------------------------------------------------

def rag_chat(user_question, history):
    if history is None:
        history = []

    # Search for context
    docs = vectorstore.similarity_search(user_question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Format the prompt
    prompt = prompt_template.format(
        context=context,
        question=user_question
    )

    # Get LLM response
    answer = llm.invoke(prompt)

    # Append as dictionaries
    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": answer})
    
    return history


# ----------------------------------------------------
# 7. GRADIO UI
# ----------------------------------------------------

print("Launching Gradio UI...")

with gr.Blocks() as demo:
    gr.Markdown("## 📄 MiniBot – RAG PDF Chatbot")

    chatbot = gr.Chatbot(label="Chat History")

    msg = gr.Textbox(
        label="Ask a question from the PDF",
        placeholder="Type your question here..."
    )

    clear = gr.Button("Clear Chat")

    msg.submit(rag_chat, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])
    

    clear.click(
        lambda: [],
        None,
        chatbot
    )

demo.launch(debug=True)

