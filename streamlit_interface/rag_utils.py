"""
RAG (Retrieval-Augmented Generation) utilities for accessing FAISS vector database.
This module provides functions to search and retrieve documents from the knowledge base.
"""

import os
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict
import faiss
from llama_cpp import Llama

logger = logging.getLogger(__name__)

# Paths
RAG_DIR = "./rag_database"
MODEL_PATH = "./models/nomic-embed-text-v1.5.Q4_K_M.gguf"

# Configuration
MAX_RESULTS = int(os.getenv("RAG_MAX_RESULTS", "5"))

# Global state
_embedding_model = None
_faiss_index = None
_all_chunks = []
_chunk_metadata = []
_rag_loaded = False


def _init_rag():
    """Initialize RAG components on first use."""
    global _embedding_model, _faiss_index, _all_chunks, _chunk_metadata, _rag_loaded
    
    if _rag_loaded:
        return
    
    try:
        # Load embedding model
        _embedding_model = Llama(
            model_path=str(MODEL_PATH),
            embedding=True,
            verbose=False,
        )
        logger.info("Embedding model loaded for RAG")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return
    
    try:
        # Load FAISS index and metadata
        if RAG_DIR.exists():
            index_path = RAG_DIR / "faiss_index.bin"
            chunks_path = RAG_DIR / "chunks.pkl"
            metadata_path = RAG_DIR / "metadata.pkl"
            
            if index_path.exists():
                _faiss_index = faiss.read_index(str(index_path))
                
                with open(chunks_path, "rb") as f:
                    _all_chunks = pickle.load(f)
                
                with open(metadata_path, "rb") as f:
                    _chunk_metadata = pickle.load(f)
                
                _rag_loaded = True
                logger.info(f"RAG database loaded. Total chunks: {len(_all_chunks)}")
            else:
                logger.warning(f"FAISS index not found at {index_path}")
        else:
            logger.warning(f"RAG directory not found at {RAG_DIR}")
    except Exception as e:
        logger.error(f"Error loading RAG database: {e}")
        _rag_loaded = False


def is_rag_available() -> bool:
    """Check if RAG is available and initialized."""
    _init_rag()
    return _rag_loaded and _embedding_model is not None and _faiss_index is not None


def search(query: str, top_k: int = MAX_RESULTS) -> List[Dict]:
    """
    Search the RAG database for relevant documents.
    
    Args:
        query: The search query string
        top_k: Number of top results to return
    
    Returns:
        List of dicts with 'text' and 'source' fields
    """
    _init_rag()
    
    if not _rag_loaded or _embedding_model is None or _faiss_index is None:
        logger.warning("RAG not available for search")
        return []
    
    try:
        # Get embedding for the query
        query_embedding = _embedding_model.embed(query)
        query_emb = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS index
        D, I = _faiss_index.search(query_emb, top_k)
        
        results = []
        for idx in I[0]:
            if 0 <= idx < len(_all_chunks):
                source = "Unknown"
                if idx < len(_chunk_metadata):
                    source = _chunk_metadata[idx].get("source", "Unknown")
                
                results.append({
                    "text": _all_chunks[idx],
                    "source": source,
                    "score": float(D[0][I[0].tolist().index(idx)])
                })
        
        logger.info(f"RAG search: {len(results)} results for query: '{query[:80]}'")
        return results
    
    except Exception as e:
        logger.error(f"Error searching RAG: {e}")
        return []


def format_search_results(results: List[Dict]) -> str:
    """
    Format RAG search results as a string for inclusion in prompts.
    
    Args:
        results: List of search results from search()
    
    Returns:
        Formatted string with results
    """
    if not results:
        return ""
    
    formatted = "Here are relevant documents from the knowledge base:\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"**Document {i}** (Source: {result['source']})\n"
        formatted += f"{result['text']}\n\n"
    
    return formatted
