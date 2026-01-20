from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import requests

from langchain_community.document_loaders import PyPDFLoader
import arxiv

# --------------------------------------------------
# App initialization
# --------------------------------------------------

app = FastAPI(
    title="Swat Chat Backend",
    description="Backend API for RAG-based chatbot on SWaT testbed",
    version="0.1.0"
)

# --------------------------------------------------
# GitHub-hosted SWaT Operation Manual (Secondary KB)
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PDF_PATH = DATA_DIR / "SWaT Operation Manual.pdf"

GITHUB_PDF_URL = (
    "https://raw.githubusercontent.com/"
    "MeeraYasmin/Swat-Chat/main/data/SWaT%20Operation%20Manual.pdf"
)

def download_swat_manual():
    """
    Download the SWaT Operation Manual from GitHub if not present locally.
    """
    if not PDF_PATH.exists():
        print("[INFO] Downloading SWaT Operation Manual from GitHub...")
        response = requests.get(GITHUB_PDF_URL, timeout=30)
        response.raise_for_status()
        PDF_PATH.write_bytes(response.content)
        print("[INFO] Download completed")

try:
    download_swat_manual()
    loader = PyPDFLoader(str(PDF_PATH))
    swat_docs = loader.load()
    print(f"[INFO] Loaded {len(swat_docs)} pages from SWaT manual")
except Exception as e:
    swat_docs = []
    print(f"[ERROR] Failed to load SWaT manual: {e}")

# --------------------------------------------------
# Request & Response Models
# --------------------------------------------------

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    timestamp: str

# --------------------------------------------------
# arXiv Retrieval (Primary Knowledge Source)
# --------------------------------------------------

def search_arxiv_swat(query: str, max_results: int = 3):
    """
    Search arXiv for SWaT and CPS-related research papers.
    Acts as the PRIMARY retrieval source.
    """

    constrained_query = (
        '(ti:"SWaT" OR abs:"SWaT" OR abs:"Secure Water Treatment") '
        'AND (abs:"cyber physical" OR abs:"industrial control" '
        'OR abs:"ICS" OR abs:"water treatment") '
        f'AND ({query})'
    )

    search = arxiv.Search(
        query=constrained_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    for paper in search.results():
        results.append({
            "title": paper.title,
            "summary": paper.summary[:400],
            "published": paper.published.date().isoformat(),
            "categories": paper.categories
        })

    return results

# --------------------------------------------------
# Chat Endpoint (Scaffold – Orchestration Layer)
# --------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Scaffold chat endpoint.

    Planned pipeline:
    1. Retrieve relevant research from arXiv (PRIMARY source)
    2. Ground results using SWaT Operation Manual (SECONDARY source)
    3. Generate response using LLM
    """

    placeholder_answer = (
        "This is a placeholder response from Swat Chat. "
        "The RAG pipeline will retrieve relevant research from arXiv "
        "as the primary source and ground it using the SWaT Operation Manual."
    )

    return ChatResponse(
        answer=placeholder_answer,
        sources=["arXiv (primary)", "SWaT Operation Manual (secondary)"],
        timestamp=datetime.utcnow().isoformat()
    )

# --------------------------------------------------
# arXiv Search Endpoint (Primary Source)
# --------------------------------------------------

@app.post("/arxiv-search")
def arxiv_search(request: ChatRequest):
    papers = search_arxiv_swat(request.query)
    return {
        "query": request.query,
        "primary_source": "arXiv",
        "results": papers
    }

# --------------------------------------------------
# Diagnostics & Health Endpoints
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/kb-info")
def kb_info():
    return {
        "documents_loaded": len(swat_docs),
        "secondary_source": "SWaT Operation Manual",
        "source_location": "GitHub (public repo)"
    }
