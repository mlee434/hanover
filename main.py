import argparse
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from openai import OpenAI
from datetime import datetime


def process_query(query: str, context: str = None, conversation_history: list = None) -> tuple[str, str]:
    """
    Process the given query and return an answer using web context and ChatGPT.
    If initial context is insufficient, performs a new search automatically.
    
    Args:
        query: The question string to process
        context: Existing context to reuse (optional)
        conversation_history: Previous conversation for context
        
    Returns:
        A tuple of (answer, context) from ChatGPT based on web context
    """
    print(f"Processing query: {query}")
    
    # Create new context if not provided
    if context is None:
        print("Gathering context from web search...")
        context = create_context(query)
        
        # Check if context creation was successful
        if context.startswith("Error") or context.startswith("No"):
            return f"Could not gather sufficient context: {context}", ""
    else:
        print("Using existing context...")
    
    print("Asking ChatGPT...\n")
    
    # Ask ChatGPT using the context
    answer, found_sufficient_info = ask_chatgpt(query, context, conversation_history)
    
    # If insufficient context was found, try a new search with a more specific query
    if not found_sufficient_info and not context.startswith("Error"):
        print("🔍 Initial context insufficient. Performing targeted search...")
        print("🧠 Generating enhanced search query...")
        
        # Use ChatGPT to create a better search query
        enhanced_query = enhance_query(query, conversation_history)
        print(f"🔎 Enhanced search query: '{enhanced_query}'")
        new_context = create_context(enhanced_query)
        
        if not new_context.startswith("Error") and not new_context.startswith("No"):
            print("Retrying with enhanced context...\n")
            answer, found_sufficient_info = ask_chatgpt(query, new_context, conversation_history)
            
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


def search(query: str) -> list:
    """
    Search the web for information related to the query.
    
    Args:
        query: The search query
        
    Returns:
        A list of organic search results from Google
    """
    params = {
        "q": query,
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": "042f51c79e8bd515a1f519c9f389240e40919b180bf6271f1a43138e4a67e1a3"
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    return results["organic_results"]


def get_site_text(url: str) -> str:
    """
    Fetch and extract text content from a web page.
    
    Args:
        url: The URL of the web page to scrape
        
    Returns:
        The text content of the page, or an error message if the request fails
    """
    try:
        # Set headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request with a timeout
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text and clean it up
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        return f"Error parsing content: {str(e)}"


def create_context(query: str) -> str:
    """
    Create context for an LLM by searching and scraping the first 10 web results.
    
    Args:
        query: The search query to find relevant information
        
    Returns:
        A concatenated string of text content from the first 10 search results
    """
    try:
        # Get search results
        search_results = search(query)
        
        if not search_results:
            return "No search results found for the query."
        
        # Extract URLs from the first 10 results
        urls = []
        for i, result in enumerate(search_results[:10]):
            if 'link' in result:
                urls.append(result['link'])
        
        if not urls:
            return "No valid URLs found in search results."
        
        # Scrape text from each URL
        context_parts = []
        for i, url in enumerate(urls, 1):
            print(f"Scraping URL {i}/{len(urls)}: {url}")
            
            page_text = get_site_text(url)
            
            # Check if scraping was successful
            if not page_text.startswith("Error"):
                # Truncate very long content to keep context manageable
                if len(page_text) > 2000:
                    page_text = page_text[:2000] + "..."
                
                context_parts.append(f"=== Source {i}: {url} ===\n{page_text}\n")
            else:
                context_parts.append(f"=== Source {i}: {url} ===\n{page_text}\n")
        
        if not context_parts:
            return "No content could be scraped from the search results."
        
        # Combine all context
        full_context = "\n".join(context_parts)
        
        # Add a header with the original query
        final_context = f"=== CONTEXT FOR QUERY: '{query}' ===\n\n{full_context}"
        
        return final_context
        
    except Exception as e:
        return f"Error creating context: {str(e)}"


def ask_chatgpt(query: str, context: str, conversation_history: list = None) -> tuple[str, bool]:
    """
    Ask ChatGPT a question using the provided context and conversation history.
    
    Args:
        query: The user's question
        context: The context information gathered from web scraping
        conversation_history: Previous conversation messages for context
        
    Returns:
        A tuple of (answer, found_sufficient_info) where found_sufficient_info indicates
        if the AI found enough information in the context to answer the question
    """
    try:
        # Initialize OpenAI client
        client = OpenAI()  # This will use the OPENAI_API_KEY environment variable
        
        # Create the prompt with context and question
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Use the context information to provide accurate, detailed answers. 

IMPORTANT: You MUST cite your sources when providing information. When you reference information from the context:
- Use citations like [Source 1], [Source 2], etc. to reference the numbered sources in the context
- Include the source citations inline with the relevant information
- At the end of your response, include a "Sources:" section with the format: "[Source X] URL" for each source you cited

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


def export_conversation(conversation_history: list, filename: str) -> bool:
    """
    Export the conversation history to a text file.
    
    Args:
        conversation_history: List of conversation messages
        filename: Name of the file to save to
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        # Ensure filename has .txt extension
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        # Format the conversation
        formatted_conversation = "HANOVER - AI Q&A Conversation Export\n"
        formatted_conversation += "=" * 50 + "\n\n"
        
        for i, message in enumerate(conversation_history):
            if message["role"] == "user":
                formatted_conversation += f"USER: {message['content']}\n\n"
            elif message["role"] == "assistant":
                formatted_conversation += f"ASSISTANT: {message['content']}\n\n"
                formatted_conversation += "-" * 40 + "\n\n"
        
        # Remove the last separator
        formatted_conversation = formatted_conversation.rstrip("-" + "\n" + " ")
        
        # Add export timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_conversation += f"\n\nExported on: {timestamp}\n"
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(formatted_conversation)
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting conversation: {str(e)}")
        return False


def interactive_chat(initial_query: str):
    """
    Run an interactive chat session with follow-up questions.
    
    Args:
        initial_query: The first question to process
    """
    conversation_history = []
    current_context = None
    
    # Process initial query
    print("=" * 60)
    print("🤖 HANOVER - AI-Powered Q&A with Web Context")
    print("=" * 60)
    print()
    
    answer, current_context = process_query(initial_query)
    print("🤖 Answer:")
    print(answer)
    print()
    
    # Add to conversation history
    conversation_history.append({"role": "user", "content": initial_query})
    conversation_history.append({"role": "assistant", "content": answer})
    
    # Interactive loop for follow-up questions
    while True:
        print("-" * 60)
        print("💬 Ask a follow-up question, or type:")
        print("   • 'new' - Search web for new topic")
        print("   • 'export' - Save conversation to file")
        print("   • 'quit' or 'exit' - End conversation")
        print("-" * 60)
        
        try:
            follow_up = input("\n❓ Your question: ").strip()
            
            if not follow_up:
                continue
                
            # Handle exit commands
            if follow_up.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Thanks for using Hanover! Goodbye!")
                break
            
            # Handle export command
            if follow_up.lower() == 'export':
                print("\n💾 Export conversation to file")
                filename = input("📄 Enter filename (without extension): ").strip()
                
                if filename:
                    if export_conversation(conversation_history, filename):
                        print(f"✅ Conversation exported to '{filename}.txt'")
                    else:
                        print("❌ Failed to export conversation")
                else:
                    print("❌ No filename provided, export cancelled")
                
                print()  # Add spacing before next prompt
                continue
            
            # Handle new topic command
            if follow_up.lower() == 'new':
                print("\n🔍 Starting fresh with new web search...")
                current_context = None
                follow_up = input("❓ What would you like to know about: ").strip()
                if not follow_up or follow_up.lower() in ['quit', 'exit', 'bye', 'q']:
                    continue
            
            print()
            
            # Process follow-up question
            answer, current_context = process_query(
                follow_up, 
                current_context, 
                conversation_history
            )
            
            print("🤖 Answer:")
            print(answer)
            print()
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": follow_up})
            conversation_history.append({"role": "assistant", "content": answer})
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using Hanover! Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")


def main():
    parser = argparse.ArgumentParser(description="Hanover - Interactive AI Q&A with web context")
    parser.add_argument(
        "--query", 
        type=str, 
        required=True, 
        help="The initial question you want to ask"
    )
    
    args = parser.parse_args()
    
    # Start interactive chat session
    interactive_chat(args.query)


if __name__ == "__main__":
    main()
