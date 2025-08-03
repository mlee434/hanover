"""
Main entry point for Hanover AI Q&A system.
"""

import argparse
from conversation import interactive_chat


def main():
    parser = argparse.ArgumentParser(description="Hanover - Interactive AI Q&A with web context and semantic search")
    parser.add_argument(
        "--query", 
        type=str, 
        required=False, 
        help="The initial question you want to ask (if not provided, you'll be prompted)"
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic search and use traditional keyword-based search"
    )
    parser.add_argument(
        "--semantic-chunks",
        type=int,
        default=8,
        help="Number of most relevant content chunks to use for semantic search (default: 8, range: 1-20)"
    )
    
    args = parser.parse_args()
    
    # Get query from command line or prompt user
    if args.query:
        query = args.query
    else:
        print("🤖 Welcome to Hanover - AI-Powered Q&A Assistant!")
        print("=" * 50)
        try:
            query = input("\n❓ What would you like to know about: ").strip()
            if not query:
                print("❌ No question provided. Exiting...")
                return
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return
        except EOFError:
            print("\n❌ No input provided. Exiting...")
            return
    
    # Determine search mode
    use_semantic = not args.no_semantic
    
    print()
    if use_semantic:
        print("🧠 Semantic Search Mode: ON")
        print("📊 Will analyze and rank content chunks by relevance")
        print(f"🎯 Using top {args.semantic_chunks} most relevant chunks")
    else:
        print("🔍 Traditional Search Mode: ON")
        print("📄 Will use all available content from search results")
    print()
    
    # Start interactive chat session
    interactive_chat(query, use_semantic, args.semantic_chunks)


if __name__ == "__main__":
    main()