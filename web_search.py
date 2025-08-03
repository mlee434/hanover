"""
Web search and scraping functionality for Hanover AI Q&A system.
"""

import os
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import asyncio
from typing import List, Dict, Tuple


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
        "api_key": os.getenv("SERPAPI_API_KEY")
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


async def get_site_text_async(url: str, session=None) -> str:
    """
    Async version of get_site_text for concurrent web scraping.
    
    Args:
        url: The URL of the web page to scrape
        session: aiohttp ClientSession (will create if not provided)
        
    Returns:
        The text content of the page, or an error message if the request fails
    """
    try:
        import aiohttp
        
        # Set headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Use provided session or create a temporary one
        if session is None:
            async with aiohttp.ClientSession() as temp_session:
                return await get_site_text_async(url, temp_session)
        
        # Make the async request with timeout
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with session.get(url, headers=headers, timeout=timeout) as response:
            response.raise_for_status()  # Raise an exception for bad status codes
            content = await response.read()
            
            # Parse the HTML content
            soup = BeautifulSoup(content, 'html.parser')
            
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
        
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


async def scrape_urls_async(urls: List[str], search_results: List[Dict]) -> List[Tuple[str, str, str]]:
    """
    Scrape multiple URLs concurrently using async.
    
    Args:
        urls: List of URLs to scrape
        search_results: Search results for getting page titles
        
    Returns:
        List of tuples: (url, page_text, title)
    """
    try:
        import aiohttp
        
        async def scrape_single_url(session, url, title, index):
            """Scrape a single URL with progress tracking"""
            try:
                print(f"🔍 Scraping URL {index}/{len(urls)}: {url}")
                page_text = await get_site_text_async(url, session)
                return (url, page_text, title)
            except Exception as e:
                print(f"⚠️  Error scraping {url}: {str(e)}")
                return (url, f"Error scraping URL: {str(e)}", title)
        
        # Create aiohttp session with appropriate settings
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)  # Limit concurrent connections
        timeout = aiohttp.ClientTimeout(total=15)  # 15 second timeout
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create tasks for all URLs
            tasks = []
            for i, url in enumerate(urls):
                title = search_results[i].get('title', f'Source {i+1}') if i < len(search_results) else f'Source {i+1}'
                task = scrape_single_url(session, url, title, i+1)
                tasks.append(task)
            
            # Execute all scraping tasks concurrently
            print("🚀 Starting concurrent web scraping...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle any exceptions
            scraped_pages = []
            for result in results:
                if isinstance(result, Exception):
                    print(f"⚠️  Scraping exception: {str(result)}")
                    scraped_pages.append(("", f"Error: {str(result)}", ""))
                else:
                    scraped_pages.append(result)
            
            print(f"✅ Completed scraping {len(scraped_pages)} URLs")
            return scraped_pages
            
    except Exception as e:
        print(f"⚠️  Error in async scraping: {str(e)}")
        # Fallback to synchronous scraping
        print("🔄 Falling back to synchronous scraping...")
        return scrape_urls_sync(urls, search_results)


def scrape_urls_sync(urls: List[str], search_results: List[Dict]) -> List[Tuple[str, str, str]]:
    """
    Fallback synchronous URL scraping.
    """
    scraped_pages = []
    for i, url in enumerate(urls):
        title = search_results[i].get('title', f'Source {i+1}') if i < len(search_results) else f'Source {i+1}'
        print(f"🔍 Scraping URL {i+1}/{len(urls)}: {url}")
        page_text = get_site_text(url)
        scraped_pages.append((url, page_text, title))
    
    return scraped_pages


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