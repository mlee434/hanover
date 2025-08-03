"""
Text processing utilities for Hanover AI Q&A system.
"""

import re
from typing import List


def chunk_text(text: str, max_chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better semantic processing.
    
    Args:
        text: The text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    
    # Try to split on paragraphs first
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            current_chunk = current_chunk[-overlap:] + paragraph
        else:
            current_chunk += ("\n\n" if current_chunk else "") + paragraph
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # If we still have chunks that are too long, split on sentences
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # Split long chunks on sentences
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            current_sentence_chunk = ""
            
            for sentence in sentences:
                if len(current_sentence_chunk) + len(sentence) > max_chunk_size and current_sentence_chunk:
                    final_chunks.append(current_sentence_chunk.strip())
                    current_sentence_chunk = sentence
                else:
                    current_sentence_chunk += (" " if current_sentence_chunk else "") + sentence
            
            if current_sentence_chunk.strip():
                final_chunks.append(current_sentence_chunk.strip())
    
    return final_chunks