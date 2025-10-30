from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os

def initialize_system():
    print("\n🔄 Initializing the system...")
    docs = load_all_documents(r"C:\Users\STIC-11\Desktop\Nchat\rag1\pdf")
    store = FaissVectorStore("faiss_store")
    
    # Check if index exists, if not build it
    if not os.path.exists(os.path.join("faiss_store", "faiss.index")):
        print("📦 Building new vector store...")
        store.build_from_documents(docs)
    else:
        print("📂 Loading existing vector store...")
        store.load()
    
    rag_search = RAGSearch()
    print("\n✅ System initialized and ready!")
    return rag_search

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_welcome():
    clear_screen()
    print("=" * 60)
    print("🏥 Welcome to the Nursing Information Assistant 🏥")
    print("=" * 60)
    print("\nI can help you with information about nursing colleges and related topics.")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("Type 'clear' to clear the screen.")
    print("-" * 60)

def main():
    print_welcome()
    rag_search = initialize_system()
    
    while True:
        try:
            print("\n👤 You:", end=" ")
            query = input().strip()
            
            if query.lower() in ['quit', 'exit']:
                print("\n👋 Thank you for using the Nursing Information Assistant. Goodbye!")
                break
            
            if query.lower() == 'clear':
                print_welcome()
                continue
                
            if not query:
                continue
            
            import time
            start_time = time.time()
            
            print("\n🤖 Assistant: ", end="")
            summary = rag_search.search_and_summarize(query, top_k=3)
            print(summary)
            
            end_time = time.time()
            response_time = end_time - start_time
            print(f"\n⏱️ Response time: {response_time:.2f} seconds")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
