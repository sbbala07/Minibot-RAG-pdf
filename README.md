📄 MiniBot – RAG PDF Chatbot

MiniBot is a Retrieval Augmented Generation (RAG) based chatbot that allows users to ask questions from a PDF document using a local LLM powered by Ollama.

This project demonstrates how modern AI applications combine:

📚 Document Retrieval (FAISS Vector Database)

🤖 Local LLM Inference (Ollama)

🔎 Semantic Search (Embeddings)

💬 Interactive UI (Gradio)


🚀 Features

=> Chat with any PDF document

=> Fully offline AI chatbot

=> Uses local LLM (llama3.2:1b) – low RAM friendly

=> Uses FAISS vector search for fast retrieval

=> Beginner friendly and lightweight

=> Privacy-safe (documents stay local)


🧠 Tech Stack

Technology   -     	Purpose

Python	     -      Core programming language

LangChain	   -      RAG pipeline orchestration

Ollama	     -      Local LLM hosting

FAISS	       -      Vector similarity search

Gradio	     -      Chat UI interface

PyPDF	       -      PDF text extraction



📂 Project Structure


Minibot/
│

├── app.py              # Main chatbot application

├── policy.pdf          # User adds their own PDF here

├── .gitignore

├── requirements.txt    # Dependency list

└── README.md

⚙️ Installation Guide

✅ Step 1 — Clone Repository
    git clone https://github.com/sbbala07/Minibot-RAG-pdf.git
    
    cd Minibot-RAG-pdf

✅ Step 2 — Install Python Dependencies
    
    python -m pip install -U langchain langchain-community langchain-ollama faiss-cpu gradio pypdf

✅ Step 3 — Install Ollama
    Download from:
    👉 https://ollama.com/download
    After installation, verify:
    
    ollama --version
    
✅ Step 4 — Download Required Models
  
    ollama pull llama3.2:1b
    ollama pull nomic-embed-text

✅ Step 5 — Add Your PDF
    Place your PDF inside project folder and rename it:
    
    policy.pdf    # Any pdf of your choice

✅ Step 6 — Run Ollama Server
    
    ollama serve

✅ Step 7 — Run Chatbot
    Open new terminal and run:
   
    python app.py

✅ Step 8 — Open Browser
    You will see:
   
    http://127.0.0.1:7860
Open it and start chatting with your PDF.

🔧 How It Works (RAG Pipeline)


    PDF →  Text Split →  Embeddings →  FAISS Vector Store
                      ↓
    User Question → Similarity Search → Context Retrieval
                      ↓
          Local LLM (Ollama)
                      ↓
            Answer Generation

📌 Why RAG?
- RAG improves LLM accuracy by:
- Preventing hallucinations
- Using real document context
- Making AI responses reliable
- Keeping data private and local

⚡ Performance Notes
- First query may take ~10 seconds (model warm-up)
- Later queries become faster
- Designed for low GPU / CPU machines

🔒 Privacy Advantage

This chatbot runs fully locally:

No cloud API usage

No data sharing

Safe for sensitive documents

🧪 Future Improvements

~ Multiple PDF support

~ Streaming responses

~ Better UI styling

~ Model fine-tuning

~ Chat memory improvement

~ Planned: Dockerization and UI improvements



👨‍💻 Author

Balachandran

AI & Data Science learner with hands-on experience building local LLM applications using LangChain, Ollama, and FAISS.


⭐ Support

If you like this project:

- Star ⭐ the repository
- Share feedback
- Suggest improvements

📜 License

This project is open-source and available for educational purposes.

🎯 Learning Outcome

This project demonstrates:
- Building real-world AI applications
- Using local LLMs
- Implementing Retrieval Augmented Generation
- Integrating vector databases
- Creating interactive AI UI

This project helped me understand practical challenges such as model latency, vector search tuning, and Gradio message formats.

