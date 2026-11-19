"""
Main entry point for the RAG system.
"""

import sys
import os
import re

# Allow running this script from either project root or the src folder.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline import pipeline
from src.retriever import get_retriever
from src.agent import create_qa_agent, query_agent


def extract_search_keywords(question: str) -> str:
    """Extract useful search terms from a natural-language question."""
    stop_words = {
        "a", "about", "an", "and", "are", "can", "does", "for", "how",
        "in", "is", "of", "on", "or", "the", "to", "treatment", "what",
        "when", "which", "with", "why", "recent", "latest",
    }
    words = re.findall(r"[A-Za-z0-9-]+", question.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 1]
    return " ".join(dict.fromkeys(keywords)) or question.strip()


def main():
    """Main application flow."""
    
    print("\n" + "="*70)
    print("AGENTIC RAG SYSTEM FOR PUBMED")
    print("="*70 + "\n")
    
    # Capture the question before optional ingestion so it can be processed automatically.
    retriever = get_retriever()
    initial_question = input("What question would you like to ask? ").strip()
    
    # Step 1: Load sample data (optional)
    setup_data = input("Would you like to load sample PubMed articles? (y/n): ").lower() == 'y'
    
    if setup_data:
        query = extract_search_keywords(initial_question)
        print(f"Searching PubMed using keywords: {query}")
        
        # Fetch and process data (faster default run size)
        documents = pipeline(query, max_articles=50)
        
        if documents:
            # Add to vector store
            retriever.vector_store.add_documents(documents)
            print(f"\n✓ Loaded {len(documents)} document chunks into vector database\n")
        else:
            print("No documents to load.\n")
    # Step 2: Create agent
    print("Initializing QA agent...")
    agent = create_qa_agent(memory_enabled=True)
    print("✓ Agent ready\n")
    
    # Step 3: Interactive QA loop
    print("="*70)
    print("INTERACTIVE QA SESSION")
    print("="*70)
    print("Ask questions about the loaded documents.")
    print("Type 'quit' to exit.\n")
    
    question = initial_question
    while True:
        if not question:
            question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        print("\n" + "-"*70)
        print("Searching and reasoning...")
        print("-"*70 + "\n")
        
        response = query_agent(agent, question)
        
        print("\n" + "-"*70)
        print("ANSWER:")
        print("-"*70)
        print(response)
        print("-"*70)

        question = ""


def demo():
    """Run a quick demo without interaction."""
    print("\n" + "="*70)
    print("AGENTIC RAG SYSTEM - DEMO MODE")
    print("="*70 + "\n")
    
    # Load sample questions
    sample_questions = [
        "What are the recent advances in deep learning?",
        "Summarize the findings about neural networks.",
        "What is the relationship between machine learning and AI?"
    ]
    
    print("Creating agent...")
    agent = create_qa_agent(memory_enabled=True)
    print("✓ Agent ready\n")
    
    for question in sample_questions:
        print("\n" + "="*70)
        print(f"Question: {question}")
        print("="*70)
        
        response = query_agent(agent, question)
        print(f"\nAnswer:\n{response}\n")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
