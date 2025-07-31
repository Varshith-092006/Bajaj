import os
import re
import json
import pickle
import hashlib
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Enhanced chunk with metadata"""
    text: str
    source_file: str
    page_number: int
    chunk_id: int
    section_type: str  # header, paragraph, table, list
    importance_score: float
    entities: List[str]

@dataclass
class ProcessingResult:
    """Result of document processing"""
    decision: str
    answer: str
    amount: Optional[str]
    clause_reference: str
    justification: str
    confidence: str
    relevant_chunks: List[DocumentChunk]
    processing_metadata: Dict

class EnhancedDocumentProcessor:
    """Enhanced document processor with improved accuracy"""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize models
        self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm_model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Patterns for entity extraction
        self.entity_patterns = {
            'age': r'\b(\d{1,3})\s*(?:year|yr|y)s?\s*old\b|\b(\d{1,3})\s*(?:M|F)\b',
            'amount': r'\$?[\d,]+(?:\.\d{2})?|\b\d+\s*(?:lakh|crore|thousand)\b',
            'duration': r'\b(\d+)\s*(?:month|year|day|week)s?\b',
            'location': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'medical_procedure': r'\b(?:surgery|operation|treatment|procedure|therapy)\b',
            'policy_terms': r'\b(?:premium|deductible|coverage|claim|benefit|exclusion)\b'
        }
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract relevant entities from text"""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(filter(None, match))
                if match:
                    entities.append(f"{entity_type}:{match}")
        
        return entities
    
    def classify_section_type(self, text: str) -> str:
        """Classify the type of document section"""
        text_lower = text.lower()
        
        # Check for headers (short text with keywords)
        if len(text.split()) < 10:
            if any(keyword in text_lower for keyword in ['section', 'clause', 'article', 'chapter']):
                return 'header'
        
        # Check for tables (structured data patterns)
        if re.search(r'\b\d+\.\d+\b|\b[A-Z]\)\b|\b\|\s*\w+\s*\|', text):
            return 'table'
        
        # Check for lists
        if re.search(r'^\s*[•\-\*]\s+|\b\d+\.\s+|\b[a-z]\)\s+', text, re.MULTILINE):
            return 'list'
        
        return 'paragraph'
    
    def calculate_importance_score(self, text: str, query: str = "") -> float:
        """Calculate importance score for a chunk"""
        score = 0.0
        text_lower = text.lower()
        
        # Base score for length (longer chunks might be more informative)
        score += min(len(text.split()) / 100, 0.3)
        
        # Score for important keywords
        important_keywords = [
            'coverage', 'benefit', 'exclusion', 'premium', 'deductible',
            'waiting period', 'pre-existing', 'claim', 'policy', 'insured',
            'amount', 'limit', 'maximum', 'minimum', 'eligible', 'condition'
        ]
        
        for keyword in important_keywords:
            if keyword in text_lower:
                score += 0.1
        
        # Score for numerical values (often important in insurance)
        if re.search(r'\b\d+\b', text):
            score += 0.2
        
        # Score for query relevance if query provided
        if query:
            query_terms = set(query.lower().split())
            text_terms = set(text_lower.split())
            overlap = len(query_terms.intersection(text_terms))
            score += (overlap / max(len(query_terms), 1)) * 0.5
        
        return min(score, 1.0)
    
    def enhanced_pdf_extraction(self, pdf_path: str) -> List[DocumentChunk]:
        """Enhanced PDF extraction with metadata"""
        chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Extract text with layout information
                text_dict = page.get_text("dict")
                page_text = self.extract_structured_text(text_dict)
                
                if not page_text.strip():
                    continue
                
                # Split into semantic chunks
                page_chunks = self.semantic_chunking(page_text)
                
                for chunk_id, chunk_text in enumerate(page_chunks):
                    if len(chunk_text.strip()) < 20:  # Skip very short chunks
                        continue
                    
                    # Extract entities
                    entities = self.extract_entities(chunk_text)
                    
                    # Classify section type
                    section_type = self.classify_section_type(chunk_text)
                    
                    # Calculate importance
                    importance = self.calculate_importance_score(chunk_text)
                    
                    chunk = DocumentChunk(
                        text=chunk_text.strip(),
                        source_file=os.path.basename(pdf_path),
                        page_number=page_num + 1,
                        chunk_id=chunk_id,
                        section_type=section_type,
                        importance_score=importance,
                        entities=entities
                    )
                    
                    chunks.append(chunk)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting from {pdf_path}: {e}")
        
        return chunks
    
    def extract_structured_text(self, text_dict: Dict) -> str:
        """Extract text while preserving structure"""
        text_parts = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                block_text = []
                for line in block["lines"]:
                    line_text = []
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            line_text.append(text)
                    if line_text:
                        block_text.append(" ".join(line_text))
                
                if block_text:
                    text_parts.append("\n".join(block_text))
        
        return "\n\n".join(text_parts)
    
    def semantic_chunking(self, text: str, max_chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Improved semantic chunking"""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            para_words = paragraph.split()
            para_length = len(para_words)
            
            # If paragraph is too long, split it by sentences
            if para_length > max_chunk_size:
                sentences = self.split_into_sentences(paragraph)
                for sentence in sentences:
                    sent_length = len(sentence.split())
                    
                    if current_length + sent_length > max_chunk_size and current_chunk:
                        # Save current chunk
                        chunks.append(' '.join(current_chunk))
                        
                        # Start new chunk with overlap
                        overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = overlap_words + sentence.split()
                        current_length = len(current_chunk)
                    else:
                        current_chunk.extend(sentence.split())
                        current_length += sent_length
            else:
                # Add whole paragraph
                if current_length + para_length > max_chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_words + para_words
                    current_length = len(current_chunk)
                else:
                    current_chunk.extend(para_words)
                    current_length += para_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def build_enhanced_index(self, chunks: List[DocumentChunk]) -> Tuple[faiss.Index, np.ndarray]:
        """Build enhanced FAISS index with weighted embeddings"""
        if not chunks:
            raise ValueError("No chunks provided")
        
        # Extract text and weights
        texts = [chunk.text for chunk in chunks]
        weights = np.array([chunk.importance_score for chunk in chunks])
        
        # Generate embeddings
        embeddings = self.embed_model.encode(texts, show_progress_bar=False)
        
        # Apply importance weighting
        weighted_embeddings = embeddings * weights.reshape(-1, 1)
        
        # Build index
        if len(embeddings) > 1000:
            nlist = min(100, len(embeddings) // 10)
            index = faiss.IndexIVFFlat(faiss.IndexFlatL2(embeddings.shape[1]), embeddings.shape[1], nlist)
            index.train(weighted_embeddings.astype('float32'))
        else:
            index = faiss.IndexFlatL2(embeddings.shape[1])
        
        index.add(weighted_embeddings.astype('float32'))
        
        return index, weighted_embeddings
    
    def enhanced_retrieval(self, query: str, index: faiss.Index, chunks: List[DocumentChunk], top_k: int = 10) -> List[DocumentChunk]:
        """Enhanced retrieval with multiple ranking factors"""
        # Get initial candidates
        query_embedding = self.embed_model.encode([query])
        scores, indices = index.search(query_embedding.astype('float32'), min(top_k * 2, len(chunks)))
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                chunk = chunks[idx]
                semantic_score = 1 / (1 + scores[0][i])  # Convert distance to similarity
                candidates.append((chunk, semantic_score))
        
        # Re-rank based on multiple factors
        reranked = []
        query_terms = set(query.lower().split())
        
        for chunk, semantic_score in candidates:
            # Lexical overlap
            chunk_terms = set(chunk.text.lower().split())
            lexical_score = len(query_terms.intersection(chunk_terms)) / max(len(query_terms), 1)
            
            # Entity relevance
            entity_score = 0
            for entity in chunk.entities:
                if any(term in entity.lower() for term in query_terms):
                    entity_score += 0.1
            entity_score = min(entity_score, 0.5)
            
            # Section type bonus
            section_bonus = 0.1 if chunk.section_type in ['table', 'list'] else 0
            
            # Combined score
            final_score = (
                semantic_score * 0.5 +
                lexical_score * 0.3 +
                entity_score * 0.1 +
                chunk.importance_score * 0.05 +
                section_bonus * 0.05
            )
            
            reranked.append((chunk, final_score))
        
        # Sort by final score and return top chunks
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in reranked[:top_k]]
    
    def build_context_aware_prompt(self, chunks: List[DocumentChunk], query: str) -> str:
        """Build context-aware prompt with chunk metadata"""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            metadata = f"[Source: {chunk.source_file}, Page: {chunk.page_number}, Type: {chunk.section_type}]"
            context_parts.append(f"Context {i+1} {metadata}:\n{chunk.text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return f"""
You are an expert insurance policy analyst with deep knowledge of policy terms, conditions, and claim processing.

POLICY CONTEXT:
{context}

QUERY: {query}

ANALYSIS INSTRUCTIONS:
1. Carefully analyze each context piece, paying attention to source and type metadata
2. Look for specific clauses, conditions, amounts, and time periods
3. Consider the hierarchy of policy terms (general vs specific conditions)
4. If multiple contexts conflict, prioritize more specific or recent information
5. Extract exact amounts, percentages, and time periods when available

DECISION FRAMEWORK:
- APPROVED: Clear coverage with no exclusions or waiting periods violated
- REJECTED: Explicit exclusion or condition not met
- CONDITIONAL: Coverage available but with specific conditions/limitations
- INFORMATION_NEEDED: Insufficient information to make determination

RESPONSE FORMAT (JSON only):
{{
  "decision": "approved/rejected/conditional/information_needed",
  "answer": "Specific answer with exact policy terms",
  "amount": "Exact coverage amount if applicable, null otherwise",
  "clause_reference": "Specific clause/section reference with page number",
  "justification": "Step-by-step reasoning based on policy analysis",
  "confidence": "high/medium/low based on clarity of policy terms",
  "key_factors": ["list", "of", "key", "decision", "factors"],
  "conditions": "Any conditions or limitations if decision is conditional"
}}
"""
    
    def process_query(self, query: str, pdf_paths: List[str]) -> ProcessingResult:
        """Process query with enhanced accuracy"""
        try:
            # Extract chunks from all documents
            all_chunks = []
            for pdf_path in pdf_paths:
                chunks = self.enhanced_pdf_extraction(pdf_path)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                raise ValueError("No content extracted from documents")
            
            # Update importance scores based on query
            for chunk in all_chunks:
                chunk.importance_score = self.calculate_importance_score(chunk.text, query)
            
            # Build enhanced index
            index, embeddings = self.build_enhanced_index(all_chunks)
            
            # Retrieve relevant chunks
            relevant_chunks = self.enhanced_retrieval(query, index, all_chunks, top_k=8)
            
            # Build context-aware prompt
            prompt = self.build_context_aware_prompt(relevant_chunks, query)
            
            # Query LLM
            response = self.llm_model.generate_content(prompt)
            result_text = response.text
            
            # Parse response
            try:
                # Clean JSON from response
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1]
                
                result_data = json.loads(result_text.strip())
                
                return ProcessingResult(
                    decision=result_data.get("decision", "unknown"),
                    answer=result_data.get("answer", "No answer provided"),
                    amount=result_data.get("amount"),
                    clause_reference=result_data.get("clause_reference", "Not specified"),
                    justification=result_data.get("justification", "No justification provided"),
                    confidence=result_data.get("confidence", "low"),
                    relevant_chunks=relevant_chunks,
                    processing_metadata={
                        "total_chunks": len(all_chunks),
                        "relevant_chunks": len(relevant_chunks),
                        "key_factors": result_data.get("key_factors", []),
                        "conditions": result_data.get("conditions")
                    }
                )
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return ProcessingResult(
                    decision="error",
                    answer=result_text,
                    amount=None,
                    clause_reference="N/A",
                    justification="Failed to parse structured response",
                    confidence="low",
                    relevant_chunks=relevant_chunks,
                    processing_metadata={"error": str(e)}
                )
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

