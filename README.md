# Hanover - AI-Powered Q&A Assistant

Hanover is a command-line Q&A tool that searches the web, processes content using semantic similarity, and generates responses with GPT-4. It maintains conversation context for follow-up questions.

## Overview

Hanover searches the web, chunks text content, ranks chunks by semantic similarity to queries, and uses GPT-4 to generate answers with source citations. It supports follow-up questions and conversation export.

## Key Features

- **Semantic Search**: Uses OpenAI embeddings to rank content chunks by relevance
- **Context Building**: Selects highest-scoring chunks from multiple sources
- **Conversational Interface**: Maintains context across follow-up questions
- **Source Citations**: Provides numbered citations with URLs
- **Async Processing**: Concurrent web scraping and embedding generation
- **Conversation Export**: Save conversations as text files
- **Configurable**: Choose between semantic and traditional search modes

## Installation & Usage

This project uses the `uv` package manager for dependency management.

```bash
# Install uv (on macOS)
brew install uv

# Install dependencies
uv sync

# Run with initial question
uv run python main.py --query "When was the iPhone 4 released?"

# Interactive mode (prompts for question)
uv run python main.py

# Traditional search mode (disable semantic search)
uv run python main.py --no-semantic

# Adjust semantic search chunk count
uv run python main.py --semantic-chunks 12
```

## Development Steps:

This project was built iteratively, starting with basic functionality and adding features:

### 1. **Basic Q&A with Context** (Foundation)
- **Technique**: Simple web search + concatenation of results
- **Implementation**: Basic SerpAPI integration, single-threaded scraping with BeautifulSoup
- **Goal**: Get a working prototype that could answer questions using web data

The initial version searched Google, scraped the first 10 results, concatenated all text, and sent it to GPT-4. This had limitations with context quality and relevance.

### 2. **Source Citations** (Credibility)
- **Technique**: Numbered source tracking and citation formatting
- **Implementation**: Modified context creation to include source numbers, updated prompts to require citations
- **Goal**: Make answers trustworthy and verifiable

Added a citation system where each source gets a number (Source 1, Source 2, etc.) and the AI cites sources inline and provides a "Sources:" section with URLs.

### 3. **Conversational Interface** (User Experience)
- **Technique**: Message history tracking and interactive loop
- **Implementation**: Conversation history array, interactive CLI with follow-up prompts
- **Goal**: Enable natural back-and-forth conversations

Built an interactive chat interface for follow-up questions while maintaining context from previous exchanges.

### 4. **Smart Re-searching** (Intelligence)
- **Technique**: LLM-powered query enhancement and insufficient context detection
- **Implementation**: GPT-4 query refinement, "INSUFFICIENT_CONTEXT" detection and retry logic
- **Goal**: Improve search when initial results are inadequate

When the AI detects insufficient context, it uses GPT-4 to enhance the search query and performs a new search.

### 5. **Conversation Export** (Utility)
- **Technique**: Structured text formatting with timestamps
- **Implementation**: Simple file I/O with conversation formatting
- **Goal**: Allow users to save and share conversations

Added ability to export conversations as formatted text files with timestamps.

### 6. **Semantic Search** (Core Innovation)
- **Technique**: Text chunking + embeddings + cosine similarity ranking
- **Implementation**: 
  - Text chunking with paragraph/sentence-aware splitting and overlap
  - OpenAI text-embedding-3-small for vector generation
  - Cosine similarity calculation for relevance ranking
  - Top-k chunk selection for context building
- **Goal**: Replace naive concatenation with intelligent content selection

Instead of using all scraped content, the system now:
- Chunks text into manageable pieces (800 chars with 100 char overlap)
- Generates embeddings for each chunk and the user query
- Uses cosine similarity to rank chunks by semantic relevance
- Selects only the most relevant chunks for context

### 7. **Async Optimization** (Performance)
- **Technique**: Concurrent web scraping and embedding generation
- **Implementation**:
  - `aiohttp` for async web scraping with connection limits
  - `AsyncOpenAI` for concurrent embedding generation
  - Semaphore-based rate limiting for API calls
  - Progress tracking for user feedback
- **Goal**: Reduce response time through parallelization

Changed the system from sequential to concurrent processing, reducing response time from 45+ seconds to 15-20 seconds.

### 8. **Code Refactoring** (Maintainability)
- **Technique**: Module separation and clean architecture
- **Implementation**: Split monolithic code into focused modules:
  - `core.py`: Main query processing logic
  - `web_search.py`: Web scraping and traditional search
  - `semantic_search.py`: Embeddings and semantic ranking
  - `ai_client.py`: LLM interactions
  - `text_processing.py`: Text utilities
  - `conversation.py`: UI and conversation management
- **Goal**: Create maintainable, testable code architecture

## Technical Architecture

### Core Technologies
- **Web Search**: SerpAPI for Google search results
- **Web Scraping**: BeautifulSoup + aiohttp for async content extraction
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Similarity**: Cosine similarity with scikit-learn
- **LLM**: GPT-4 for query enhancement and answer generation
- **Async**: asyncio + aiohttp for concurrent processing

### Data Flow
1. **Search**: Query → SerpAPI → URLs
2. **Scrape**: URLs → aiohttp → Raw HTML → BeautifulSoup → Clean Text
3. **Chunk**: Text → Paragraph/sentence-aware chunking with overlap
4. **Embed**: Chunks → AsyncOpenAI → Vector embeddings (concurrent)
5. **Rank**: Query embedding + Chunk embeddings → Cosine similarity → Top-k selection
6. **Generate**: Selected chunks + Query → GPT-4 → Cited answer

### Performance Optimizations
- **Concurrent Processing**: 10+ URLs scraped simultaneously
- **Semaphore Limiting**: Max 10 concurrent embedding requests
- **Chunking**: Paragraph and sentence boundary awareness
- **Connection Pooling**: aiohttp connector with connection limits
- **Fallback Mechanisms**: Degradation to sync processing

## What I Would Do With More Time

1. **Caching System**: Redis/SQLite caching for search results and embeddings
2. **Error Handling**: Error recovery and user error messages
3. **Configuration File**: YAML config for chunk sizes, similarity thresholds, API keys
4. **Multiple Search Engines and LLMs**: Bing, DuckDuckGo, Claude, Gemini, etc.
5. **Agent Framework**: Instead of hard coded functions, use Temporal + custom code to create distinct agents that can make choices on tool calling. Using something like Temporal would also provide resiliency and force the code structure to be much stricter on modularity and separation of concerns.


