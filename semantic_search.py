"""
Semantic search and embeddings functionality for Hanover AI Q&A system.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import asyncio
from typing import List, Dict, Tuple
from text_processing import chunk_text
from web_search import search, scrape_urls_async


def get_embedding(text: str, client: OpenAI = None) -> np.ndarray:
    """
    Generate vector embedding for the given text using OpenAI's embedding model.
    
    Args:
        text: The text to embed
        client: OpenAI client instance (will create if not provided)
        
    Returns:
        Numpy array containing the embedding vector
    """
    try:
        if client is None:
            client = OpenAI()
        
        # Clean and truncate text if too long (embedding models have limits)
        cleaned_text = text.strip()
        if len(cleaned_text) > 8000:  # Conservative limit
            cleaned_text = cleaned_text[:8000] + "..."
        
        response = client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-3-small"
        )
        
        return np.array(response.data[0].embedding)
        
    except Exception as e:
        print(f"⚠️  Error generating embedding: {str(e)}")
        # Return zero vector as fallback
        return np.zeros(1536)  # text-embedding-3-small has 1536 dimensions


async def get_embedding_async(text: str, client=None) -> np.ndarray:
    """
    Async version of get_embedding for parallel processing.
    
    Args:
        text: The text to embed
        client: AsyncOpenAI client instance (will create if not provided)
        
    Returns:
        Numpy array containing the embedding vector
    """
    try:
        # Import AsyncOpenAI here to avoid issues if not available
        from openai import AsyncOpenAI
        
        if client is None:
            client = AsyncOpenAI()
        
        # Clean and truncate text if too long (embedding models have limits)
        cleaned_text = text.strip()
        if len(cleaned_text) > 8000:  # Conservative limit
            cleaned_text = cleaned_text[:8000] + "..."
        
        response = await client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-3-small"
        )
        
        return np.array(response.data[0].embedding)
        
    except Exception as e:
        print(f"⚠️  Error generating embedding: {str(e)}")
        # Return zero vector as fallback
        return np.zeros(1536)  # text-embedding-3-small has 1536 dimensions


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    try:
        # Reshape for sklearn if needed
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return max(0.0, similarity)  # Ensure non-negative
        
    except Exception as e:
        print(f"⚠️  Error calculating similarity: {str(e)}")
        return 0.0


def semantic_search(query: str, content_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Find the most semantically similar content chunks to the query.
    
    Args:
        query: The user's query
        content_chunks: List of dictionaries with 'text', 'embedding', 'source', 'chunk_id'
        top_k: Number of top similar chunks to return
        
    Returns:
        List of most relevant chunks with similarity scores
    """
    try:
        client = OpenAI()
        
        # Get query embedding
        query_embedding = get_embedding(query, client)
        
        # Calculate similarities
        similarities = []
        for i, chunk in enumerate(content_chunks):
            if 'embedding' in chunk and chunk['embedding'] is not None:
                similarity = calculate_similarity(query_embedding, chunk['embedding'])
                similarities.append({
                    'chunk': chunk,
                    'similarity': similarity,
                    'index': i
                })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
        
    except Exception as e:
        print(f"⚠️  Error in semantic search: {str(e)}")
        # Fallback to returning first few chunks
        return [{'chunk': chunk, 'similarity': 0.0, 'index': i} 
                for i, chunk in enumerate(content_chunks[:top_k])]


async def generate_embeddings_async(text_chunks: List[Dict]) -> List[Dict]:
    """
    Generate embeddings for multiple text chunks concurrently using async.
    
    Args:
        text_chunks: List of chunk dictionaries with text and metadata
        
    Returns:
        List of chunks with embeddings added
    """
    try:
        from openai import AsyncOpenAI
        
        async_client = AsyncOpenAI()
        
        async def generate_single_embedding(chunk_data):
            """Generate embedding for a single chunk"""
            try:
                embedding = await get_embedding_async(chunk_data['text'], async_client)
                return {
                    **chunk_data,
                    'embedding': embedding
                }
            except Exception as e:
                print(f"⚠️  Error generating embedding for chunk: {str(e)}")
                return {
                    **chunk_data,
                    'embedding': np.zeros(1536)  # Fallback zero vector
                }
        
        # Limit concurrent requests to respect API rate limits
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
        
        async def limited_generate_embedding(chunk_data):
            async with semaphore:
                return await generate_single_embedding(chunk_data)
        
        # Generate all embeddings concurrently
        print("🔄 Starting concurrent embedding generation...")
        
        # Create tasks for all chunks
        tasks = [limited_generate_embedding(chunk) for chunk in text_chunks]
        
        # Process with progress tracking
        all_chunks = []
        completed = 0
        
        # Use asyncio.as_completed for progress tracking
        for task in asyncio.as_completed(tasks):
            chunk_with_embedding = await task
            all_chunks.append(chunk_with_embedding)
            completed += 1
            
            # Show progress
            if completed % 5 == 0 or completed == len(text_chunks):
                print(f"📊 Generated embeddings: {completed}/{len(text_chunks)}")
        
        await async_client.close()  # Clean up the client
        return all_chunks
        
    except Exception as e:
        print(f"⚠️  Error in async embedding generation: {str(e)}")
        # Fallback to sync generation
        print("🔄 Falling back to synchronous embedding generation...")
        return generate_embeddings_sync(text_chunks)


def generate_embeddings_sync(text_chunks: List[Dict]) -> List[Dict]:
    """
    Fallback synchronous embedding generation.
    """
    client = OpenAI()
    all_chunks = []
    
    for i, chunk_data in enumerate(text_chunks, 1):
        try:
            embedding = get_embedding(chunk_data['text'], client)
            all_chunks.append({
                **chunk_data,
                'embedding': embedding
            })
            
            if i % 5 == 0 or i == len(text_chunks):
                print(f"📊 Generated embeddings: {i}/{len(text_chunks)} (sync)")
                
        except Exception as e:
            print(f"⚠️  Error generating embedding: {str(e)}")
            all_chunks.append({
                **chunk_data,
                'embedding': np.zeros(1536)
            })
    
    return all_chunks


async def create_semantic_context(query: str, max_chunks: int = 8) -> Tuple[str, List[Dict]]:
    """
    Create context using semantic search to find the most relevant information.
    
    Args:
        query: The search query to find relevant information
        
    Returns:
        A tuple of (context_string, metadata_list) where metadata contains relevance info
    """
    try:
        print("🧠 Using semantic search for enhanced context retrieval...")
        
        # Get search results
        search_results = search(query)
        
        if not search_results:
            return "No search results found for the query.", []
        
        # Extract URLs from the first 10 results
        urls = []
        for result in search_results[:10]:
            if 'link' in result:
                urls.append(result['link'])
        
        if not urls:
            return "No valid URLs found in search results.", []
        
        # Scrape and chunk content using async for better performance
        all_text_chunks = []  # Store chunks with metadata for parallel processing
        
        print(f"🌐 Scraping {len(urls)} URLs concurrently...")
        scraped_pages = await scrape_urls_async(urls, search_results)
        
        # Process scraped content into chunks (limit to first 100 chunks per website)
        for i, (url, page_text, title) in enumerate(scraped_pages, 1):
            if not page_text.startswith("Error"):
                # Create chunks from this page
                chunks = chunk_text(page_text)
                
                # Limit to first 100 chunks per website to control processing time and costs
                chunks_processed = 0
                for j, text_chunk in enumerate(chunks):
                    if len(text_chunk.strip()) > 50:  # Only process substantial chunks
                        all_text_chunks.append({
                            'text': text_chunk,
                            'source': url,
                            'source_id': i,
                            'chunk_id': j,
                            'title': title
                        })
                        chunks_processed += 1
                        
                        # Stop after 100 chunks per website
                        if chunks_processed >= 100:
                            print(f"📏 Limited to first {chunks_processed} chunks for {url}")
                            break
        
        if not all_text_chunks:
            return "No content could be processed for semantic search.", []
        
        print(f"🧠 Generating embeddings for {len(all_text_chunks)} chunks using async...")
        
        # Second pass: generate embeddings using async
        all_chunks = await generate_embeddings_async(all_text_chunks)
        
        if not all_chunks:
            return "No content could be processed for semantic search.", []
        
        print(f"🔍 Analyzing {len(all_chunks)} content chunks for semantic relevance...")
        
        # Find most relevant chunks using semantic search
        relevant_chunks = semantic_search(query, all_chunks, top_k=max_chunks)
        
        # Group chunks by source and build clean context
        source_mapping = {}  # Maps original source_id to new source number
        context_parts = []
        metadata = []
        current_source_num = 1
        
        for result in relevant_chunks:
            chunk = result['chunk']
            similarity = result['similarity']
            original_source_id = chunk['source_id']
            
            # Map to a clean source number for citation
            if original_source_id not in source_mapping:
                source_mapping[original_source_id] = current_source_num
                current_source_num += 1
            
            source_num = source_mapping[original_source_id]
            
            # Add content with clean source numbering
            context_parts.append(
                f"=== Source {source_num} ===\n"
                f"URL: {chunk['source']}\n"
                f"Title: {chunk['title']}\n"
                f"{chunk['text']}\n"
            )
        
        # Create metadata for sources section (unique sources only)
        unique_sources = {}
        relevance_scores = {}
        
        for result in relevant_chunks:
            chunk = result['chunk']
            similarity = result['similarity']
            original_source_id = chunk['source_id']
            source_num = source_mapping[original_source_id]
            
            if source_num not in unique_sources:
                unique_sources[source_num] = {
                    'url': chunk['source'],
                    'title': chunk['title'],
                    'source_num': source_num
                }
                relevance_scores[source_num] = []
            
            relevance_scores[source_num].append(similarity)
        
        # Build final metadata
        for source_num in sorted(unique_sources.keys()):
            source_info = unique_sources[source_num]
            avg_relevance = sum(relevance_scores[source_num]) / len(relevance_scores[source_num])
            max_relevance = max(relevance_scores[source_num])
            
            metadata.append({
                'source_num': source_num,
                'url': source_info['url'],
                'title': source_info['title'],
                'avg_similarity': avg_relevance,
                'max_similarity': max_relevance,
                'chunk_count': len(relevance_scores[source_num])
            })
        
        # Combine all context
        full_context = "\n".join(context_parts)
        
        # Add header with semantic search info
        final_context = (
            f"=== SEMANTIC CONTEXT FOR QUERY: '{query}' ===\n"
            f"Found {len(relevant_chunks)} most relevant chunks from {len(unique_sources)} sources\n\n"
            f"{full_context}"
        )
        
        return final_context, metadata
        
    except Exception as e:
        print(f"⚠️  Error in semantic context creation: {str(e)}")
        # Fallback to original method
        from web_search import create_context
        return create_context(query), []