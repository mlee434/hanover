"""
Conversation management and UI functionality for Hanover AI Q&A system.
"""

from datetime import datetime


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


def interactive_chat(initial_query: str, use_semantic: bool = True, semantic_chunks: int = 8):
    """
    Run an interactive chat session with follow-up questions.
    
    Args:
        initial_query: The first question to process
        use_semantic: Whether to use semantic search
    """
    from core import process_query
    
    conversation_history = []
    current_context = None
    
    # Process initial query
    print("=" * 60)
    if use_semantic:
        print("🧠 HANOVER - AI-Powered Q&A with Semantic Search")
    else:
        print("🤖 HANOVER - AI-Powered Q&A with Web Context")
    print("=" * 60)
    print()
    
    answer, current_context = process_query(initial_query, use_semantic=use_semantic, semantic_chunks=semantic_chunks)
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
                conversation_history,
                use_semantic,
                semantic_chunks
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