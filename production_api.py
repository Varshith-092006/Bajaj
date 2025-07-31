import os
import logging
import pickle
import hashlib
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import aiofiles
import requests
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pdfplumber
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Document Processing API",
    description="API for processing and querying insurance documents using LLM",
    version="1.0.0"
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

# Configuration
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

# Initialize models
embed_model = None
gemini_model = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class DocumentResponse(BaseModel):
    content: str
    similarity_score: float
    source: str

def cosine_similarity(a, b):
    """Pure Python cosine similarity implementation"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def load_models():
    """Load embedding and LLM models"""
    global embed_model, gemini_model
    
    try:
        # Load embedding model
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
        
        # Configure Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini model configured successfully")
        else:
            logger.warning("GEMINI_API_KEY not found")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def save_to_cache(data, cache_key, cache_type):
    """Save data to cache"""
    try:
        cache_file = CACHE_DIR / f"{cache_type}_{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved {cache_type} to cache: {cache_key}")
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")

def load_from_cache(cache_key, cache_type):
    """Load data from cache"""
    try:
        cache_file = CACHE_DIR / f"{cache_type}_{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded {cache_type} from cache: {cache_key}")
            return data
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
    return None

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

def download_and_extract_text_optimized(url: str) -> str:
    """Download PDF and extract text with caching"""
    try:
        # Create cache key from URL
        cache_key = hashlib.md5(url.encode()).hexdigest()
        
        # Check cache first
        cached_text = load_from_cache(cache_key, "pdf_text")
        if cached_text:
            return cached_text
        
        # Download PDF
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file
        temp_file = CACHE_DIR / f"temp_{cache_key}.pdf"
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        # Extract text
        text = extract_text_from_pdf(str(temp_file))
        
        # Cache the text
        save_to_cache(text, cache_key, "pdf_text")
        
        # Clean up temp file
        temp_file.unlink(missing_ok=True)
        
        return text
        
    except Exception as e:
        logger.error(f"Error downloading/extracting PDF: {e}")
        raise

def semantic_chunking(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Create semantic chunks from text"""
    try:
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error in semantic chunking: {e}")
        return [text]

def build_vector_index_pure_python(chunks: List[str], embed_model) -> tuple:
    """Build vector index using pure Python"""
    try:
        # Generate embeddings
        vectors = embed_model.encode(chunks, show_progress_bar=False, batch_size=32)
        
        # Create cache key
        chunks_text = "".join(chunks)
        cache_key = hashlib.md5(chunks_text.encode()).hexdigest()
        
        # Cache the vectors and chunks
        cache_data = {'vectors': vectors, 'chunks': chunks}
        save_to_cache(cache_data, cache_key, "vector_index")
        
        return vectors, chunks
        
    except Exception as e:
        logger.error(f"Error building vector index: {e}")
        raise

def search_chunks_pure_python(vectors, query: str, embed_model, all_chunks: List[str], top_k: int = 5) -> List[Dict]:
    """Search chunks using pure Python cosine similarity"""
    try:
        # Generate query embedding
        query_vector = embed_model.encode([query])
        
        # Calculate similarities
        similarities = []
        for i, vector in enumerate(vectors):
            similarity = cosine_similarity(query_vector[0], vector)
            similarities.append((similarity, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Get top results
        results = []
        for similarity, idx in similarities[:top_k]:
            if idx < len(all_chunks):
                results.append({
                    'content': all_chunks[idx],
                    'similarity_score': float(similarity),
                    'source': f'chunk_{idx}'
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching chunks: {e}")
        raise

def process_documents(urls: List[str]) -> tuple:
    """Process documents and build search index"""
    try:
        all_chunks = []
        
        for url in urls:
            logger.info(f"Processing document: {url}")
            text = download_and_extract_text_optimized(url)
            chunks = semantic_chunking(text)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            raise ValueError("No text extracted from documents")
        
        # Build vector index
        vectors, chunks = build_vector_index_pure_python(all_chunks, embed_model)
        
        return vectors, chunks
        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise

def generate_llm_response(query: str, relevant_chunks: List[Dict]) -> str:
    """Generate LLM response using Gemini"""
    try:
        if not gemini_model:
            return "LLM not configured. Please set GEMINI_API_KEY environment variable."
        
        # Prepare context
        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
        
        # Create prompt
        prompt = f"""Based on the following insurance document context, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the information is not available in the context, please state that clearly."""

        # Generate response
        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        return f"Error generating response: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_models()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Document Processing API",
        "status": "running",
        "pdf_library": "pdfplumber",
        "vector_search": "pure-python"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gemini_configured = os.getenv('GEMINI_API_KEY') is not None
    return {
        "status": "healthy",
        "pdf_library": "pdfplumber",
        "vector_search": "pure-python",
        "gemini_configured": gemini_configured
    }

@app.post("/process-documents")
async def process_documents_endpoint(urls: List[str]):
    """Process documents and build search index"""
    try:
        if not urls:
            raise HTTPException(status_code=400, detail="No URLs provided")
        
        vectors, chunks = process_documents(urls)
        
        return {
            "message": "Documents processed successfully",
            "num_chunks": len(chunks),
            "vectors_shape": vectors.shape
        }
        
    except Exception as e:
        logger.error(f"Error in process_documents_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(request: QueryRequest):
    """Query documents using semantic search and LLM"""
    try:
        # Load cached vectors and chunks
        cache_files = list(CACHE_DIR.glob("vector_index_*.pkl"))
        if not cache_files:
            raise HTTPException(status_code=400, detail="No processed documents found. Please process documents first.")
        
        # Load the most recent cache
        latest_cache = max(cache_files, key=lambda x: x.stat().st_mtime)
        cache_data = load_from_cache(latest_cache.stem.replace("vector_index_", ""), "vector_index")
        
        if not cache_data:
            raise HTTPException(status_code=500, detail="Failed to load cached data")
        
        vectors = cache_data['vectors']
        chunks = cache_data['chunks']
        
        # Search for relevant chunks
        relevant_chunks = search_chunks_pure_python(
            vectors, request.query, embed_model, chunks, request.top_k
        )
        
        # Generate LLM response
        llm_response = generate_llm_response(request.query, relevant_chunks)
        
        return {
            "query": request.query,
            "relevant_chunks": relevant_chunks,
            "llm_response": llm_response
        }
        
    except Exception as e:
        logger.error(f"Error in query_documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        file_path = CACHE_DIR / file.filename
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Extract text
        text = extract_text_from_pdf(str(file_path))
        chunks = semantic_chunking(text)
        
        # Build vector index
        vectors, processed_chunks = build_vector_index_pure_python(chunks, embed_model)
        
        # Clean up uploaded file
        file_path.unlink(missing_ok=True)
        
        return {
            "message": "PDF processed successfully",
            "filename": file.filename,
            "num_chunks": len(processed_chunks),
            "text_length": len(text)
        }
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

