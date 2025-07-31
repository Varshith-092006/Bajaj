import os
import time
import json
import pickle
import hashlib
import fitz  # PyMuPDF
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 📦 Load environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Cache directory
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(text):
    """Generate cache key from text content"""
    return hashlib.md5(text.encode()).hexdigest()

def load_from_cache(cache_key, cache_type="embeddings"):
    """Load data from cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_type}_{cache_key}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_to_cache(data, cache_key, cache_type="embeddings"):
    """Save data to cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_type}_{cache_key}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def clean_text(text):
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Remove special characters that might interfere with processing
    text = text.replace('\x00', '').replace('\ufffd', '')
    return text

# 📄 Enhanced PDF extraction with better error handling
def extract_texts_from_folder(folder_path):
    """Extract text from all PDFs in folder with enhanced processing"""
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            try:
                doc = fitz.open(os.path.join(folder_path, filename))
                text_parts = []
                
                for page_num, page in enumerate(doc):
                    try:
                        # Try different extraction methods for better accuracy
                        text = page.get_text("text")
                        if not text.strip():  # If text extraction fails, try dict method
                            text_dict = page.get_text("dict")
                            text = extract_text_from_dict(text_dict)
                        text_parts.append(text)
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {filename}: {e}")
                        continue
                
                full_text = clean_text("\n".join(text_parts))
                texts[filename] = full_text
                logger.info(f"Extracted {len(full_text)} characters from {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                continue
    
    return texts

def extract_text_from_dict(text_dict):
    """Extract text from PyMuPDF dict format"""
    text_parts = []
    for block in text_dict.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
    return " ".join(text_parts)

# 🧩 Improved chunking strategy
def chunk_text_semantic(text, chunk_size=512, overlap=64):
    """Improved chunking that tries to preserve sentence boundaries"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > chunk_size and current_chunk:
            # Save current chunk
            chunks.append('. '.join(current_chunk) + '.')
            
            # Start new chunk with overlap
            overlap_sentences = current_chunk[-overlap//10:] if len(current_chunk) > overlap//10 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks

# 🧠 Optimized vector search with caching
def build_faiss_index_cached(text_chunks, embed_model):
    """Build FAISS index with caching support"""
    # Create cache key from all chunks
    combined_text = "\n".join(text_chunks)
    cache_key = get_cache_key(combined_text)
    
    # Try to load from cache
    cached_data = load_from_cache(cache_key, "faiss_index")
    if cached_data:
        logger.info("Loading FAISS index from cache")
        return cached_data['index'], cached_data['vectors']
    
    logger.info("Building new FAISS index")
    vectors = embed_model.encode(text_chunks, show_progress_bar=True)
    
    # Use more efficient index for larger datasets
    if len(vectors) > 1000:
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(vectors.shape[1]), vectors.shape[1], min(100, len(vectors)//10))
        index.train(vectors)
    else:
        index = faiss.IndexFlatL2(vectors.shape[1])
    
    index.add(vectors)
    
    # Save to cache
    cache_data = {'index': index, 'vectors': vectors}
    save_to_cache(cache_data, cache_key, "faiss_index")
    
    return index, vectors

# 🔍 Enhanced retrieval with re-ranking
def search_index_enhanced(index, query, embed_model, all_chunks, top_k=10, rerank_top_k=5):
    """Enhanced search with re-ranking for better relevance"""
    # Initial retrieval
    query_vector = embed_model.encode([query])
    scores, indices = index.search(query_vector, top_k)
    
    # Get candidate chunks
    candidates = [(all_chunks[i], scores[0][j]) for j, i in enumerate(indices[0]) if i < len(all_chunks)]
    
    # Simple re-ranking based on query term overlap
    query_terms = set(query.lower().split())
    reranked = []
    
    for chunk, score in candidates:
        chunk_terms = set(chunk.lower().split())
        overlap_score = len(query_terms.intersection(chunk_terms)) / len(query_terms)
        combined_score = (1 - score) * 0.7 + overlap_score * 0.3  # Combine semantic and lexical similarity
        reranked.append((chunk, combined_score))
    
    # Sort by combined score and return top results
    reranked.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in reranked[:rerank_top_k]]

# 🧾 Improved prompt engineering
def build_enhanced_prompt(context, question):
    """Enhanced prompt with better instructions for accuracy"""
    return f"""
You are an expert insurance policy analyst. Analyze the provided policy excerpts to answer the specific question.

POLICY EXCERPTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Base your answer ONLY on the provided policy excerpts
2. If information is not available in the excerpts, state "Information not available in provided excerpts"
3. Quote the exact clause or section when possible
4. Provide a clear, concise answer followed by justification

RESPONSE FORMAT (JSON):
{{
  "decision": "approved/rejected/conditional/information_needed",
  "answer": "Clear, specific answer based on policy",
  "amount": "Applicable amount if relevant, otherwise null",
  "clause_reference": "Exact clause or section reference",
  "justification": "Detailed explanation of how the decision was reached",
  "confidence": "high/medium/low"
}}
"""

# 🤖 Enhanced Gemini query with retry logic
async def query_gemini_async(prompt, retries=3):
    """Async Gemini query with improved error handling"""
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                wait_time = (2 ** attempt) * 30  # Exponential backoff
                logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    raise RuntimeError(f"Gemini API failed after {retries} retries: {e}")
                await asyncio.sleep(5)
    
    raise RuntimeError("Gemini API failed after all retries")

def process_single_query(query, index, embed_model, all_chunks):
    """Process a single query synchronously"""
    try:
        # Enhanced retrieval
        context_chunks = search_index_enhanced(index, query, embed_model, all_chunks, top_k=10, rerank_top_k=5)
        context_text = "\n---\n".join(context_chunks)
        
        # Build enhanced prompt
        prompt = build_enhanced_prompt(context_text, query)
        
        # Query Gemini (sync version for now)
        result = query_gemini_sync(prompt)
        
        return {
            "query": query,
            "result": result,
            "context_used": len(context_chunks)
        }
    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}")
        return {
            "query": query,
            "result": f"Error: {str(e)}",
            "context_used": 0
        }

def query_gemini_sync(prompt, retries=3):
    """Synchronous version of Gemini query"""
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                wait_time = (2 ** attempt) * 30
                logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)
            else:
                logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    raise RuntimeError(f"Gemini API failed after {retries} retries: {e}")
                time.sleep(5)
    
    raise RuntimeError("Gemini API failed after all retries")

# 🚀 Main optimized logic
def main():
    """Main function with optimizations"""
    start_time = time.time()
    
    # Configuration
    folder_path = "./bajaj_docs"
    
    # Load embedding model (cached automatically by sentence-transformers)
    logger.info("Loading embedding model...")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Extract documents
    logger.info("Extracting documents...")
    documents = extract_texts_from_folder(folder_path)
    
    if not documents:
        raise RuntimeError("No documents found or extracted!")
    
    # Process all documents with improved chunking
    logger.info("Processing and chunking documents...")
    all_chunks = []
    chunk_metadata = []
    
    for filename, text in documents.items():
        chunks = chunk_text_semantic(text, chunk_size=512, overlap=64)
        logger.info(f"📄 {filename}: {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                'filename': filename,
                'chunk_id': i,
                'length': len(chunk)
            })
    
    logger.info(f"✅ Total Chunks: {len(all_chunks)}")
    
    # Build optimized index
    logger.info("Building FAISS index...")
    index, vectors = build_faiss_index_cached(all_chunks, embed_model)
    
    # Test queries
    test_queries = [
        "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?",
        "What is the waiting period for cataract surgery?",
        "Are medical expenses for organ donor covered?",
        "What is the No Claim Discount offered?",
        "Is there benefit for preventive health check-ups?",
        "How does the policy define a Hospital?",
        "What is the coverage for AYUSH treatments?",
        "Are there sub-limits on room rent and ICU charges?"
    ]
    
    # Process queries with parallel processing for better performance
    logger.info("Processing queries...")
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_query = {
            executor.submit(process_single_query, query, index, embed_model, all_chunks): query 
            for query in test_queries
        }
        
        for future in future_to_query:
            result = future.result()
            results.append(result)
    
    # Display results
    logger.info("\n📤 Processing Results:")
    for result in results:
        print(f"\n🔍 Query: {result['query']}")
        print(f"📄 Context chunks used: {result['context_used']}")
        print(f"🤖 Response: {result['result']}")
        print("-" * 80)
    
    total_time = time.time() - start_time
    logger.info(f"\n⏱️ Total processing time: {total_time:.2f} seconds")
    logger.info(f"📊 Average time per query: {total_time/len(test_queries):.2f} seconds")

if __name__ == "__main__":
    main()

