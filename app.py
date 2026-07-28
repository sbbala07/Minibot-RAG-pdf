# ----------------------------------------------------
# RAG PDF Chatbot using Ollama + LangChain + FAISS + Gradio
# Multi- PDF Support
# ----------------------------------------------------

import gradio as gr

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate


# Tools ready before any PDF arrives (It doesn't change)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

embeddings =  OllamaEmbeddings(model = "nomic-embed-text")

llm = OllamaLLM(model="llama3.2:1b")  # Small LLM to avoid GPU/RAM issues

vectorstore = None  # Empty shelf- waiting for first pdf

# ----------------------------------------------------
# 2. PROMPT TEMPLATE
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
# 3. PDF PROCESSING FUNCTION
# ----------------------------------------------------

def process_pdf(files):
    global vectorstore
    
    if files is None:
        return "⚠️ No files uploaded."
    
    # files is a list — loop through each uploaded file
    processed = []
    for file in files:
        # Extract the file path correctly
        file_path = file.name if hasattr(file, 'name') else file
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            vectorstore.add_documents(chunks)
        
        # Get clean filename and remove duplicate .pdf extension if present
        filename = file_path.split('/')[-1].split(chr(92))[-1]
        if filename.endswith('.pdf.pdf'):
            filename = filename[:-4]  # Remove the extra .pdf. Take the string but remove the last 4 characters.
        processed.append(f"{filename} — {len(chunks)} chunks")
    
    return "✅ Processed:\n" + "\n".join(processed)

# ----------------------------------------------------
# 4. RAG CHAT FUNCTION
# ----------------------------------------------------

def rag_chat(user_question, history):
    if history is None:
        history = []

    # Guard — no PDF uploaded yet
    if vectorstore is None:
        history.append({"role": "assistant", "content": "⚠️ Please upload a PDF first before asking questions."})
        return history   # Stop the function there itself if file not uploaded

    # Build conversational context from recent history
    recent_history = history[-6:]  # Get last 3 conversation, 6 lines with question and answer
    history_text = ""              # Empty string
    for message in recent_history:  
        role = message["role"].upper() # Get role user/assistant and upper for LLM readability
        content = message["content"]   # Get content
        history_text += f"{role}: {content}\n\n"  # Role with content

    # Combine history with current question for better retrieval
    search_query = history_text + f"USER: {user_question}"

    # Search using full conversational context
    docs = vectorstore.similarity_search(search_query, k=3)
    
    context = "\n\n".join(doc.page_content for doc in docs)

    # Extract citations from retrieved chunks
    citations = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown") # Safe access — returns default if key missing instead of crashing. Unknown- if not available
        source = source.split('/')[-1].split(chr(92))[-1]  # Extract filename only
        page = doc.metadata.get("page", "?")
        citation = f"• {source} — Page {page + 1}" # Metadata page index starts at 0, humans count from 1
        if citation not in citations: # Prevents duplicate sources when multiple chunks come from same page
            citations.append(citation)

    # Format the prompt
    prompt = prompt_template.format(
        context=context,
        question=user_question
    )

    # Get LLM response
    answer = llm.invoke(prompt)

    citation_text = "\n\n📄 Sources:\n" + "\n".join(citations)
    full_answer = answer + citation_text

    # Append as dictionaries
    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": full_answer})
    
    return history

# ----------------------------------------------------
# 5. GRADIO UI
# ----------------------------------------------------

print("Launching Gradio UI...")

with gr.Blocks() as demo:
    gr.Markdown("## 📄 MiniBot – RAG PDF Chatbot")
    
    # PDF Upload Section
    with gr.Row():
        file_input = gr.File(
            label="Upload PDF(s)",
            file_types=[".pdf"],
            file_count="multiple"  # Can select multiple file at once & upload together
        )
        upload_btn = gr.Button("Process PDF")
    
    status_box = gr.Textbox(
        label="Upload Status",
        interactive=False  # read only system msg
    )
    
    # Chat Section
    chatbot = gr.Chatbot(label="Chat History")
    
    msg = gr.Textbox(
        label="Ask a question from the PDF",
        placeholder="Type your question here..."
    )
    
    clear = gr.Button("Clear Chat")
    
    # Wiring
    upload_btn.click(process_pdf, [file_input], [status_box])
    msg.submit(rag_chat, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])
    clear.click(lambda: [], None, chatbot)

demo.launch(debug=True)
