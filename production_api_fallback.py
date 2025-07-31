import os
import time
import json
import pickle
import hashlib
import requests
import tempfile
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Global variables
embed_model = None
document_cache = {}
index_cache = {}

# Cache configuration
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Environment validation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")

# Try to import PDF libraries
try:
    import fitz  # PyMuPDF
    PDF_LIBRARY = "pymupdf"
    logger.info("Using PyMuPDF for PDF processing")
except ImportError:
    try:
        import pdfplumber
        PDF_LIBRARY = "pdfplumber"
        logger.info("Using pdfplumber for PDF processing")
    except ImportError:
        PDF_LIBRARY = None
        logger.error("No PDF library available")

class QueryRequest(BaseModel):
    query: str
    documents: Optional[List[str]] = None
    use_cache: Optional[bool] = True
    top_k: Optional[int] = 5

class RunRequest(BaseModel):
    documents: List[str]
    questions: List[str]

class QueryResponse(BaseModel):
    decision: str
    answer: str
    amount: Optional[str] = None
    clause_reference: str
    justification: str
    confidence: str
    processing_time: float

class RunResponse(BaseModel):
    answers: List[str]

def get_cache_key(text):
    """Generate cache key from text content"""
    return hashlib.md5(text.encode()).hexdigest()

def load_from_cache(cache_key, cache_type="embeddings"):
    """Load data from cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_type}_{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
    return None

def save_to_cache(data, cache_key, cache_type="embeddings"):
    """Save data to cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_type}_{cache_key}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_file}: {e}")

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    text = text.encode("utf-8", "replace").decode("utf-8", "ignore")
    text = ' '.join(text.split())
    text = text.replace('\x00', '').replace('\ufffd', '')
    return text

def extract_text_with_pymupdf(pdf_path: str) -> str:
    """Extract text using PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(doc):
            try:
                text = page.get_text("text")
                if not text.strip():
                    text_dict = page.get_text("dict")
                    text = extract_text_from_dict(text_dict)
                text_parts.append(text)
            except Exception as e:
                logger.warning(f"Error extracting page {page_num}: {e}")
                continue
        
        doc.close()
        return clean_text("\n".join(text_parts))
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        raise

def extract_text_with_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber"""
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception as e:
                    logger.warning(f"Error extracting page: {e}")
                    continue
        
        return clean_text("\n".join(text_parts))
    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {e}")
        raise

def extract_text_from_dict(text_dict):
    """Extract text from PyMuPDF dict format"""
    text_parts = []
    for block in text_dict.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
    return " ".join(text_parts)

def download_and_extract_text_optimized(url: str) -> str:
    """Enhanced PDF text extraction with better error handling"""
    try:
        # Check cache first
        url_hash = get_cache_key(url)
        cached_text = load_from_cache(url_hash, "pdf_text")
        if cached_text:
            logger.info(f"Using cached text for {url}")
            return cached_text
        
        # Download PDF with timeout
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            raise ValueError(f"Failed to download: {url} (Status: {response.status_code})")
        
        # Extract text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        # Use appropriate PDF library
        if PDF_LIBRARY == "pymupdf":
            full_text = extract_text_with_pymupdf(tmp_path)
        elif PDF_LIBRARY == "pdfplumber":
            full_text = extract_text_with_pdfplumber(tmp_path)
        else:
            raise ValueError("No PDF library available")
        
        # Cache the result
        save_to_cache(full_text, url_hash, "pdf_text")
        
        # Cleanup
        os.unlink(tmp_path)
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error processing PDF {url}: {e}")
        raise ValueError(f"Failed to process PDF: {str(e)}")

def chunk_text_semantic(text, chunk_size=512, overlap=64):
    """Improved chunking that preserves sentence boundaries"""
    if not text:
        return []
    
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
            overlap_sentences = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return [chunk for chunk in chunks if chunk.strip()]

def build_faiss_index_optimized(chunks, embed_model):
    """Build optimized FAISS index with caching"""
    if not chunks:
        raise ValueError("No chunks provided for indexing")
    
    combined_text = "\n".join(chunks)
    cache_key = get_cache_key(combined_text)
    
    cached_data = load_from_cache(cache_key, "faiss_index")
    if cached_data:
        logger.info("Using cached FAISS index")
        return cached_data['index'], cached_data['vectors']
    
    logger.info(f"Building FAISS index for {len(chunks)} chunks")
    vectors = embed_model.encode(chunks, show_progress_bar=False, batch_size=32)
    
    if len(vectors) > 1000:
        nlist = min(100, len(vectors) // 10)
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(vectors.shape[1]), vectors.shape[1], nlist)
        index.train(vectors)
    else:
        index = faiss.IndexFlatL2(vectors.shape[1])
    
    index.add(vectors)
    
    cache_data = {'index': index, 'vectors': vectors}
    save_to_cache(cache_data, cache_key, "faiss_index")
    
    return index, vectors

def search_chunks_enhanced(index, query, embed_model, all_chunks, top_k=5):
    """Enhanced search with re-ranking"""
    if not all_chunks:
        return []
    
    q_vec = embed_model.encode([query])
    scores, indices = index.search(q_vec, min(top_k * 2, len(all_chunks)))
    
    candidates = []
    for j, i in enumerate(indices[0]):
        if i < len(all_chunks):
            candidates.append((all_chunks[i], scores[0][j]))
    
    query_terms = set(query.lower().split())
    reranked = []
    
    for chunk, score in candidates:
        chunk_terms = set(chunk.lower().split())
        overlap_score = len(query_terms.intersection(chunk_terms)) / max(len(query_terms), 1)
        combined_score = (1 / (1 + score)) * 0.7 + overlap_score * 0.3
        reranked.append((chunk, combined_score))
    
    reranked.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in reranked[:top_k]]

def build_enhanced_prompt(context, query):
    """Enhanced prompt for better decision making"""
    return f"""
You are an expert insurance policy analyst. Analyze the provided policy excerpts to answer the specific question.

POLICY EXCERPTS:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Base your answer ONLY on the provided policy excerpts
2. If information is not available, state "Information not available in provided excerpts"
3. Quote exact clauses when possible
4. Provide clear decision and justification

RESPONSE FORMAT (JSON only, no markdown):
{{
  "decision": "approved/rejected/conditional/information_needed",
  "answer": "Clear, specific answer based on policy",
  "amount": "Applicable amount if relevant, otherwise null",
  "clause_reference": "Exact clause or section reference",
  "justification": "Detailed explanation of how the decision was reached",
  "confidence": "high/medium/low"
}}
"""

def query_gemini_optimized(prompt, retries=3):
    """Optimized Gemini query with better error handling"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return clean_text(response.text)
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                wait_time = min((2 ** attempt) * 10, 60)
                logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)
            else:
                logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    raise HTTPException(status_code=500, detail=f"Gemini API failed: {str(e)}")
                time.sleep(2)
    
    raise HTTPException(status_code=500, detail="Gemini API failed after all retries")

def process_documents(documents: List[str]) -> tuple:
    """Process documents and return chunks and index"""
    global embed_model
    
    if embed_model is None:
        logger.info("Loading embedding model...")
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    all_chunks = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_url = {executor.submit(download_and_extract_text_optimized, url): url for url in documents}
        
        for future in future_to_url:
            url = future_to_url[future]
            try:
                text = future.result(timeout=120)
                if text:
                    chunks = chunk_text_semantic(text, chunk_size=512, overlap=64)
                    all_chunks.extend(chunks)
                    logger.info(f"Processed {url}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to process {url}: {e}")
                continue
    
    if not all_chunks:
        raise HTTPException(status_code=400, detail="No valid chunks found from documents")
    
    index, vectors = build_faiss_index_optimized(all_chunks, embed_model)
    
    return all_chunks, index

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    global embed_model
    logger.info("Starting up LLM Document Processing API...")
    logger.info(f"PDF Library: {PDF_LIBRARY}")
    logger.info("Loading embedding model...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("API ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# FastAPI app with lifespan management
app = FastAPI(
    title="LLM Document Processing API",
    version="2.0.0",
    description="Optimized API for processing insurance documents with LLM",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "gemini_configured": bool(GEMINI_API_KEY),
        "pdf_library": PDF_LIBRARY
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a single query against documents"""
    start_time = time.time()
    
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="No documents provided")
        
        all_chunks, index = process_documents(request.documents)
        
        relevant_chunks = search_chunks_enhanced(
            index, request.query, embed_model, all_chunks, top_k=request.top_k
        )
        
        context = "\n---\n".join(relevant_chunks)
        prompt = build_enhanced_prompt(context, request.query)
        result = query_gemini_optimized(prompt)
        
        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1]
            
            response_data = json.loads(result.strip())
            processing_time = time.time() - start_time
            
            return QueryResponse(
                decision=response_data.get("decision", "unknown"),
                answer=response_data.get("answer", "No answer provided"),
                amount=response_data.get("amount"),
                clause_reference=response_data.get("clause_reference", "Not specified"),
                justification=response_data.get("justification", "No justification provided"),
                confidence=response_data.get("confidence", "low"),
                processing_time=processing_time
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            processing_time = time.time() - start_time
            
            return QueryResponse(
                decision="error",
                answer=result,
                clause_reference="N/A",
                justification="Failed to parse structured response",
                confidence="low",
                processing_time=processing_time
            )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_submission(request: RunRequest):
    """Original API endpoint for compatibility"""
    try:
        all_chunks, index = process_documents(request.documents)
        
        results = []
        for question in request.questions:
            try:
                relevant_chunks = search_chunks_enhanced(
                    index, question, embed_model, all_chunks, top_k=5
                )
                context = "\n---\n".join(relevant_chunks)
                prompt = build_enhanced_prompt(context, question)
                result = query_gemini_optimized(prompt)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}")
                results.append(f"Error: {str(e)}")
        
        return RunResponse(answers=results)
        
    except Exception as e:
        logger.error(f"Error in run_submission: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Document Processing API v2.0",
        "status": "running",
        "pdf_library": PDF_LIBRARY,
        "endpoints": {
            "health": "/health",
            "query": "/api/v1/query",
            "run": "/api/v1/hackrx/run"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("production_api_fallback:app", host="0.0.0.0", port=port, reload=False) 