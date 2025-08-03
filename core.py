"""
Core query processing functionality for Hanover AI Q&A system.
"""

import asyncio
from ai_client import ask_chatgpt, enhance_query
from web_search import create_context
from semantic_search import create_semantic_context


def process_query(query: str, context: str = None, conversation_history: list = None, use_semantic: bool = True, semantic_chunks: int = 8) -> tuple[str, str]:
    """
    Process the given query and return an answer using web context and ChatGPT.
    If initial context is insufficient, performs a new search automatically.
    
    Args:
        query: The question string to process
        context: Existing context to reuse (optional)
        conversation_history: Previous conversation for context
        use_semantic: Whether to use semantic search (default: True)
        
    Returns:
        A tuple of (answer, context) from ChatGPT based on web context
    """
    print(f"Processing query: {query}")
    
    # Create new context if not provided
    if context is None:
        if use_semantic:
            print("🧠 Gathering context using semantic search...")
            context, semantic_metadata = asyncio.run(create_semantic_context(query, semantic_chunks))
        else:
            print("Gathering context from web search...")
            context = create_context(query)
            semantic_metadata = []
        
        # Check if context creation was successful
        if context.startswith("Error") or context.startswith("No"):
            return f"Could not gather sufficient context: {context}", ""
    else:
        print("Using existing context...")
        semantic_metadata = []
    
    print("Asking ChatGPT...\n")
    
    # Ask ChatGPT using the context
    answer, found_sufficient_info = ask_chatgpt(query, context, conversation_history, semantic_metadata)
    
    # If insufficient context was found, try a new search with a more specific query
    if not found_sufficient_info and not context.startswith("Error"):
        print("🔍 Initial context insufficient. Performing targeted search...")
        print("🧠 Generating enhanced search query...")
        
        # Use ChatGPT to create a better search query
        enhanced_query_text = enhance_query(query, conversation_history)
        print(f"🔎 Enhanced search query: '{enhanced_query_text}'")
        
        if use_semantic:
            new_context, new_metadata = asyncio.run(create_semantic_context(enhanced_query_text, semantic_chunks))
        else:
            new_context = create_context(enhanced_query_text)
            new_metadata = []
        
        if not new_context.startswith("Error") and not new_context.startswith("No"):
            print("Retrying with enhanced context...\n")
            answer, found_sufficient_info = ask_chatgpt(query, new_context, conversation_history, new_metadata)
            
            # If we got a better answer, use the new context
            if found_sufficient_info:
                context = new_context
            else:
                # Clean up the INSUFFICIENT_CONTEXT prefix for final answer
                if answer.startswith("INSUFFICIENT_CONTEXT:"):
                    answer = answer.replace("INSUFFICIENT_CONTEXT:", "").strip()
                    answer = f"I searched multiple sources but couldn't find comprehensive information about your question. Here's what I found:\n\n{answer}"
        else:
            # Clean up the INSUFFICIENT_CONTEXT prefix
            if answer.startswith("INSUFFICIENT_CONTEXT:"):
                answer = answer.replace("INSUFFICIENT_CONTEXT:", "").strip()
                answer = f"I couldn't find comprehensive information about your question. Here's what I found in the available sources:\n\n{answer}"
    
    return answer, context