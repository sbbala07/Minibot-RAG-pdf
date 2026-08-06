# PROJECT_CONTEXT.md — Minibot RAG PDF Chatbot
> Last updated: August 2026  
> Purpose: Enable any AI assistant to continue this project without reading the full conversation history.

---

## 1. PROJECT OVERVIEW

**Project Name:** Minibot — RAG PDF Chatbot  
**GitHub:** https://github.com/sbbala07/Minibot-RAG-pdf.git  
**Type:** Portfolio project for AI Engineer / Data Scientist career transition  
**Target Market:** Paris, France (European AI job market)  
**Current Phase:** 7-day sprint — Days 1–4 complete, Days 5–7 remaining

### Developer Profile
- **Name:** Bala (Sbbala07)
- **Background:** Electrical Engineering + MBA (Marketing & Finance)
- **Experience:** 15+ years Sales & Marketing; currently in Business Development for AI/DS education
- **Technical Level:** Intermediate Python — builds projects but learning deeply
- **Goal:** Transition to Data Scientist / AI Engineer role in Paris
- **Learning Style:** Concept-first, question-answer method. Must understand every line before writing it. Never copy-paste without understanding.
- **French:** Learning (FrenchPod101 subscriber) — bilingual README planned

---

## 2. ARCHITECTURE

### Core RAG Pipeline
```
PDF Upload
    ↓
PyPDFLoader — extracts text page by page → Document objects (page_content + metadata)
    ↓
RecursiveCharacterTextSplitter — chunk_size=500, chunk_overlap=50
    ↓
OllamaEmbeddings (nomic-embed-text) — converts chunks to vectors
    ↓
FAISS vectorstore — stores vectors in clusters for similarity search
    ↓
User asks question
    ↓
Last 3 conversation exchanges (6 messages) + current question → search_query
    ↓
FAISS similarity_search(search_query, k=3) — retrieves top 3 chunks
    ↓
Extract page_content → context (for LLM)
Extract metadata (source, page) → citations (for user display)
    ↓
ChatPromptTemplate.format(context, question) → prompt
    ↓
OllamaLLM (llama3.2:1b) → generates answer
    ↓
answer + citation_text → full_answer displayed to user
```

### Two Interfaces (Both Functional)
```
app.py   → Gradio UI (localhost:7860) — for local demo
api.py   → FastAPI backend (localhost:8000) — for production/integration
```

### File Structure
```
Minibot/
├── app.py              ← Gradio UI + RAG logic (Days 1-3)
├── api.py              ← FastAPI backend (Day 4)
├── requirements.txt    ← Python dependencies
├── README.md           ← Current README (to be overhauled Day 7)
├── Steps from scratch.txt ← Bala's personal notes
└── uploaded_pdfs/      ← Created at runtime by api.py (gitignored)
```

---

## 3. IMPLEMENTATION STATUS

### ✅ COMPLETED — Days 1–4

#### Day 1 — Multi-PDF Upload (app.py)
- Removed hardcoded `policy.pdf`
- Added `gr.File(file_count="multiple")` upload component
- `process_pdf(files)` function — loops through uploaded files
- `vectorstore = None` at startup — accumulation strategy (add, not replace)
- Guard: returns warning if no PDF uploaded before chat
- Fix: double `.pdf.pdf` extension cleaned from display
- `global vectorstore` pattern used correctly

#### Day 2 — Source Citations (app.py)
- Citations extracted from `doc.metadata` after FAISS retrieval
- `doc.metadata.get("source", "Unknown")` — safe access with default
- Filename extracted from full temp path using split
- Page number displayed as `page + 1` (metadata is 0-indexed)
- Duplicate citations filtered with `if citation not in citations`
- Citations appended below answer: `📄 Sources:\n• filename — Page N`

#### Day 3 — Conversation Memory (app.py)
- Sliding window: `history[-6:]` = last 3 exchanges
- History converted to `"ROLE: content\n\n"` format
- `search_query = history_text + f"USER: {user_question}"` 
- FAISS searches with full context → understands follow-up questions
- `user_question` (not `search_query`) still passed to prompt template
- History appended AFTER answer (not before) — avoids duplicate signal

#### Day 4 — FastAPI Backend (api.py)
- Three endpoints: `GET /health`, `POST /upload`, `POST /chat`
- Pydantic `ChatRequest(question: str, history: list = [])` validates input
- `UploadFile` + `shutil.copyfileobj` saves PDFs to `uploaded_pdfs/` folder
- CORS middleware: `allow_origins=["*"]` (dev only)
- API is stateless — frontend manages conversation history
- Returns JSON: `{"answer": "...", "sources": [...]}`
- Auto-docs at `localhost:8000/docs` (Swagger UI)
- Uvicorn: `host="0.0.0.0", port=8000`

---

### 🔲 PENDING — Days 5–7

#### Day 5 — Docker
- Create `Dockerfile` for api.py
- Create `docker-compose.yml` (app + ollama service)
- Must run with single command: `docker-compose up`
- Key interview point: "runs anywhere, not just my laptop"
- Note: Ollama in Docker needs `OLLAMA_NUM_GPU=0` on Bala's machine

#### Day 6 — Evaluation Set + Retrieval Quality Indicator
- Create 15–20 Q&A pairs from Policy.pdf manually
- Measure retrieval accuracy: how many correct answers / total
- Add retrieval quality indicator to `/chat` response:
  ```json
  {"answer": "...", "sources": [...], "retrieval_score": "high"}
  ```
- Score based on FAISS distance (not LLM confidence — unreliable)
- Document eval results in README
- Note: LLM confidence scores were discussed and REJECTED — misleading

#### Day 7 — Bilingual README + Architecture Diagram
- Full README overhaul
- English + French (bilingual — Paris market signal)
- Include: architecture diagram, eval results, setup instructions, demo GIF
- Remove `debug=True` from Gradio launch

---

## 4. DESIGN DECISIONS & RATIONALE

| Decision | What | Why |
|---|---|---|
| Accumulate vectorstore | Add new PDFs to existing FAISS index | Richer knowledge base vs replace (loses history) |
| Sliding window memory | Last 6 messages only | Avoids context window overflow, reduces noise |
| Separate search_query vs user_question | History+question for FAISS, question only for LLM prompt | Different jobs: retrieval context vs generation focus |
| Two separate models | nomic-embed-text + llama3.2:1b | Specialist > generalist; embedding model optimised for similarity |
| `page + 1` for citations | Display page N+1 | Metadata is 0-indexed; humans count from 1 |
| `.get("source", "Unknown")` | Safe metadata access | Prevents KeyError if metadata missing |
| Reject LLM confidence scores | Not implemented | LLMs produce unreliable confidence; misleads users |
| FAISS over Pinecone/Weaviate | Local vector store | No API cost, works offline, appropriate for portfolio |
| `global vectorstore` | Share state across functions | Python scoping — function-local variable would disappear |
| `exist_ok=True` | `os.makedirs` | Prevents crash if folder already exists on restart |
| `"wb"` file mode | Binary write for PDFs | PDFs are binary files; text mode corrupts them |
| FastAPI over Flask | Production API layer | Async support, auto-docs, Pydantic validation, industry standard |

---

## 5. KNOWN ISSUES & WORKAROUNDS

### GPU/RAM Memory Conflict (Windows, Ollama)
**Problem:** `nomic-embed-text` fails to load — CUDA out of memory or CPU buffer allocation failure  
**Root cause:** Ollama tries to load embedding model on GPU; GPU already occupied by other processes  
**Current workaround:**
```powershell
# In PowerShell terminal before running api.py:
$env:OLLAMA_NUM_GPU=0
python api.py
```
**Also tried (didn't work):** `os.environ["CUDA_VISIBLE_DEVICES"] = ""` — Ollama is a separate process, ignores Python env  
**Permanent fix for Day 5:** Docker with `OLLAMA_NUM_GPU=0` baked into docker-compose environment  
**Alternative:** Switch to `all-minilm` (45MB) instead of `nomic-embed-text` (273MB)

### Double Extension (policy.pdf.pdf)
**Problem:** Gradio temp files sometimes append `.pdf` to already `.pdf` filenames  
**Fix implemented:** `filename[:-4]` if `filename.endswith('.pdf.pdf')`  
**Status:** Fixed in app.py display; api.py unaffected (uses `file.filename` directly)

### Ollama Port Conflict
**Problem:** `Error: listen tcp 127.0.0.1:11434: bind: Only one usage` when running `ollama serve`  
**Meaning:** Ollama already running — this is fine, not an error  
**Workflow:** Check system tray first; only run `ollama serve` if not already running

### `langchain-community` Deprecation Warning
**Message:** "`langchain-community` is being sunset"  
**Impact:** None currently — still works  
**Future fix:** Migrate to standalone packages (Day 7 cleanup or Enterprise project)

---

## 6. CODING CONVENTIONS

### Style
- Clear section comments: `# ---- SECTION NAME ----`
- Inline comments on every non-obvious line (Bala must understand each line for interviews)
- Snake_case for variables and functions
- Descriptive variable names: `search_query` not `q`, `full_answer` not `ans`

### Patterns Used
```python
# Global state pattern
vectorstore = None          # top level
def process_pdf(files):
    global vectorstore      # always first line when modifying global

# Safe metadata access
source = doc.metadata.get("source", "Unknown")  # never doc.metadata["source"]

# Windows path extraction
filename = path.split('/')[-1].split(chr(92))[-1]  # handles both / and \

# Sliding window
recent_history = history[-6:]  # last 3 exchanges = 6 messages
```

### API Response Format (api.py)
```json
// /health
{"status": "healthy", "model": "llama3.2:1b"}

// /upload  
{"message": "PDF processed successfully", "filename": "...", "chunks_added": 316}

// /chat
{"answer": "...", "sources": ["Policy.pdf — Page 31", "Policy.pdf — Page 4"]}

// /chat error
{"error": "No PDF uploaded yet. Please upload a PDF first."}
```

---

## 7. LOCAL DEVELOPMENT SETUP

### Prerequisites
- Python 3.10+ with venv
- Ollama installed and running
- Models pulled: `ollama pull nomic-embed-text` + `ollama pull llama3.2:1b`

### Run Gradio App (app.py)
```bash
# Terminal 1 — ensure Ollama running
ollama serve   # skip if already running

# Terminal 2 — run app
cd C:\Minibot
.venv\Scripts\activate
python app.py
# Open: http://localhost:7860
```

### Run FastAPI Backend (api.py)
```powershell
# PowerShell — must set GPU env first
$env:OLLAMA_NUM_GPU=0
python api.py
# API docs: http://localhost:8000/docs
```

### Dependencies (requirements.txt — needs update)
```
gradio
langchain
langchain-community
langchain-ollama
langchain-core
langchain-text-splitters
faiss-cpu
pypdf
fastapi
uvicorn
python-multipart
pydantic
```
> Note: requirements.txt is outdated — missing fastapi, uvicorn, python-multipart. Update before Day 5 Docker build.

---

## 8. INTERVIEW PREPARATION NOTES

Bala has explicitly requested to understand every line for interviews. Key talking points built during this sprint:

### RAG Pipeline Explanation
> "When a PDF is uploaded, the system splits it into 500-character chunks with 50-character overlap. Chunking is necessary because LLMs have a context window limit and smaller focused chunks produce better quality embeddings. Each chunk is converted to a vector by nomic-embed-text — a model specialised for semantic similarity. Vectors are stored in FAISS which organises them into clusters for fast retrieval. At query time, the question vector is compared against chunk vectors via similarity search, returning the top 3 most relevant chunks. Those chunks are passed as context to llama3.2 which generates the answer. Retrieval and generation are two separate, specialised jobs."

### Key Technical Terms Bala Knows
- Hallucination, Grounding, Scope control
- Context window, Chunking, Overlap, Embeddings, Vector, Similarity search
- FAISS clustering, k (retrieval count), Sliding window memory
- Pydantic validation, CORS, REST endpoints, JSON, Stateless API
- `async def`, Binary file mode, Global scope

### Production Awareness Talking Points
- "For production I would replace Ollama with Mistral API — one line change"
- "Mistral is Paris-based — strong French language support for European enterprise"
- "I used Ollama locally to eliminate API costs during development"
- "Docker ensures it runs anywhere, not just my laptop"
- "k=3 is tunable — too low misses information, too high introduces noise"
- "I tested different chunk sizes against an eval set to justify 500"

---

## 9. BROADER CAREER CONTEXT

### Project Sequence (Agreed Roadmap)
1. **Minibot** (current) — foundation RAG skills + production habits
2. **Enterprise Knowledge Assistant** — flagship project, multi-format (PDF/DOCX/web), FastAPI, PostgreSQL, Docker, auth
3. **AI Sales Copilot** — leverages Bala's 15+ years sales background (unique differentiator)
4. **Multi-Agent Business Analyst** — LangGraph/CrewAI
5. **MLOps wrap** — Docker, CI/CD, monitoring on best project
6. **Mistral fine-tuning / French NLP** — Paris market signal

### Paris Market Signals to Build Into Projects
- Mistral AI (Paris-based) — use their models in production references
- Hugging Face ecosystem — Transformers, PEFT, LoRA
- Bilingual documentation (English + French)
- Business problem framing (Bala's MBA advantage)
- Production-ready systems (not just demos)

---

## 10. NEXT IMMEDIATE STEPS

When continuing this project, proceed to **Day 5 — Docker**:

1. Update `requirements.txt` with missing packages
2. Create `Dockerfile` for `api.py`
3. Create `docker-compose.yml` with app + ollama services
4. Add `OLLAMA_NUM_GPU=0` to docker-compose environment
5. Test: `docker-compose up` → hit `localhost:8000/docs`
6. Commit: `"Day 5: Dockerize FastAPI backend"`

**Teaching method to follow:**
- Always concept before code
- Ask Bala to explain each concept before introducing it
- Ask Bala to design logic in plain English before writing code
- Correct answers before moving forward
- Never give code without Bala first attempting an explanation
- Same pace throughout — no rushing

---

*This document was generated from a full conversation history between Bala and Claude (Anthropic). The project is actively in development as of August 2026.*
