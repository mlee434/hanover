"""
AI/LLM client functionality for Hanover AI Q&A system.
"""

from openai import OpenAI
from typing import List, Dict


def ask_chatgpt(query: str, context: str, conversation_history: list = None, semantic_metadata: List[Dict] = None) -> tuple[str, bool]:
    """
    Ask ChatGPT a question using the provided context and conversation history.
    
    Args:
        query: The user's question
        context: The context information gathered from web scraping
        conversation_history: Previous conversation messages for context
        semantic_metadata: Metadata about semantic search results with relevance scores
        
    Returns:
        A tuple of (answer, found_sufficient_info) where found_sufficient_info indicates
        if the AI found enough information in the context to answer the question
    """
    try:
        # Initialize OpenAI client
        client = OpenAI()  # This will use the OPENAI_API_KEY environment variable
        
        # System prompt for semantic search (same as traditional for user-facing content)
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Use the context information to provide accurate, detailed answers. The context has been intelligently selected using AI semantic search for maximum relevance.

IMPORTANT CITATION REQUIREMENTS:
- Use citations like [Source 1], [Source 2], etc. to reference the numbered sources in the context
- Include the source citations inline with the relevant information  
- At the end of your response, include a "Sources:" section with the format: "[Source X] URL" for each source you cited
- Do NOT include technical details like relevance scores or chunk information in your main answer

Example Sources section format:
Sources:
[Source 1] https://example1.com
[Source 3] https://example3.com
[Source 5] https://example5.com

CRITICAL: If the context doesn't contain enough information to provide a comprehensive answer to the question, you MUST start your response with the exact phrase "INSUFFICIENT_CONTEXT:" followed by your explanation of what information is missing. Then provide whatever partial information you can find in the context."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please answer the question based on the context provided above. Remember to:
1. Cite your sources using [Source 1], [Source 2], etc. when referencing information
2. Include a "Sources:" section at the end with the format "[Source X] URL" for each source you cited"""
        
        # Build messages list with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})
        
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        
        # Extract the response
        answer = response.choices[0].message.content
        
        # Add semantic search technical details to sources section if available
        if semantic_metadata and not answer.startswith("INSUFFICIENT_CONTEXT:"):
            # Extract the sources section if it exists
            if "Sources:" in answer:
                parts = answer.split("Sources:")
                main_answer = parts[0].strip()
                sources_section = "Sources:" + parts[1]
                
                # Add semantic search info to sources section
                semantic_info = "\n\n🧠 **Semantic Search Technical Details:**\n"
                for meta in semantic_metadata:
                    semantic_info += f"• Source {meta['source_num']}: {meta['chunk_count']} chunks, "
                    semantic_info += f"avg relevance: {meta['avg_similarity']:.2f}, "
                    semantic_info += f"max relevance: {meta['max_similarity']:.2f}\n"
                
                answer = main_answer + "\n\n" + sources_section + semantic_info
            else:
                # If no sources section, add semantic info at the end
                semantic_info = "\n\n🧠 **Semantic Search Analysis:**\n"
                semantic_info += f"• Analyzed {sum(meta['chunk_count'] for meta in semantic_metadata)} content chunks from {len(semantic_metadata)} sources\n"
                avg_score = sum(meta['avg_similarity'] for meta in semantic_metadata) / len(semantic_metadata)
                max_score = max(meta['max_similarity'] for meta in semantic_metadata)
                semantic_info += f"• Relevance scores: avg {avg_score:.2f}, max {max_score:.2f}\n"
                
                answer += semantic_info
        
        # Check if the AI found sufficient information
        found_sufficient_info = not answer.startswith("INSUFFICIENT_CONTEXT:")
        
        return answer, found_sufficient_info
        
    except Exception as e:
        return f"Error calling ChatGPT API: {str(e)}\n\nPlease make sure you have set the OPENAI_API_KEY environment variable.", False


def enhance_query(original_query: str, conversation_history: list = None) -> str:
    """
    Use ChatGPT to create an enhanced search query based on conversation context.
    
    Args:
        original_query: The user's original question
        conversation_history: Previous conversation for context
        
    Returns:
        An enhanced search query optimized for finding relevant information
    """
    try:
        client = OpenAI()
        
        # Build context from conversation history
        conversation_context = ""
        if conversation_history:
            # Get the last few exchanges for context
            recent_history = conversation_history[-6:]  # Last 3 exchanges (user + assistant)
            for msg in recent_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content'][:200]}...\n"
        
        system_prompt = """You are an expert at formulating search queries. Your task is to create an optimized search query that will find the most relevant information to answer a user's question.

Consider:
- The specific information needed to answer the question
- Any context from the conversation that might help narrow the search
- Keywords that would appear in authoritative sources
- Alternative phrasings or technical terms

Return ONLY the enhanced search query, nothing else."""
        
        user_prompt = f"""Original question: {original_query}

Conversation context:
{conversation_context if conversation_context else "No prior conversation context."}

Create an enhanced search query that would find the most relevant and comprehensive information to answer this question. Focus on finding authoritative, detailed sources."""
        
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        enhanced_query = response.choices[0].message.content.strip()
        
        # Remove quotes if ChatGPT wrapped the query in them
        if enhanced_query.startswith('"') and enhanced_query.endswith('"'):
            enhanced_query = enhanced_query[1:-1]
        if enhanced_query.startswith("'") and enhanced_query.endswith("'"):
            enhanced_query = enhanced_query[1:-1]
            
        return enhanced_query
        
    except Exception as e:
        # Fallback to simple enhancement if API call fails
        print(f"⚠️  Query enhancement failed: {str(e)}")
        return f"{original_query} detailed explanation information"