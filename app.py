from src.data_loader import load_all_documents
from src.vectorstore import ChromaVectorStore
from src.search import RAGSearch
import os
import re
import time


# 🧹 --- CLEANER FUNCTION ---
def clean_model_output(text: str) -> str:
    """Strip persona, logs, and system text from LLM output."""
    if not text:
        return ""
    if not isinstance(text, str):
        try:
            # if dict or list, convert safely
            text = str(text)
        except Exception:
            return ""

    # Remove known unwanted parts
    text = re.sub(r"(?s)Persona\s*&\s*Purpose:.*?End of system instructions\.?", "", text)
    text = re.sub(r"(?s)Behavioral\s*Rules\s*/\s*Constraints:.*?End of system instructions\.?", "", text)
    text = re.sub(r"(?s)Style\s*and\s*Formatting:.*?End of system instructions\.?", "", text)
    text = re.sub(r"(?s)Safety\s*and\s*Tone:.*?End of system instructions\.?", "", text)

    # Remove specific noisy lines
    noisy_lines = [
        r" Assistant:.*",
        r"\[INFO\].*",
        r"Querying vector store.*",
        r"Based on the following context.*",
        r"Context:.*",
        r"End of system instructions.*",
        r" Response time:.*",
        r"-{3,}",
    ]
    for pattern in noisy_lines:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Keep only "Answer:" section if exists
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    # Preserve any "Sources:" block if present
    if "Sources:" in text:
        parts = text.split("Sources:")
        main_text = parts[0].strip()
        sources = "Sources:" + parts[-1].strip()
        text = f"{main_text}\n\n{sources}"

    # Clean extra spaces/newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


# ⚙️ --- SYSTEM INITIALIZATION ---
def initialize_system():
    print("\n Initializing the system...")
    docs = load_all_documents(r"C:\Users\hamsa\OneDrive\Desktop\my proj\Nursing chatbot\NCH_gemini\Nchatbot\pdf")
    store = ChromaVectorStore("chromaDB")
    # Load and check collection size; rebuild if empty (handles .gitkeep case)
    store.load()
    try:
        count = store.collection.count() if hasattr(store.collection, "count") else 0
    except Exception:
        count = 0
    if count == 0:
        print(" Building new vector store (ChromaDB)...")
        store.build_from_documents(docs)
    else:
        print(" Loaded existing ChromaDB vector store...")

    rag_search = RAGSearch()
    print("\n System initialized and ready!")
    return rag_search


# 🧭 --- DISPLAY HELPERS ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_welcome():
    clear_screen()
    print("=" * 60)
    print(" Welcome to the Nursing Information Assistant ")
    print("=" * 60)
    print("\nI can help you with information about nursing colleges and related topics.")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("Type 'clear' to clear the screen.")
    print("-" * 60)


# 💬 --- MAIN CHAT LOOP ---
def main():
    print_welcome()
    rag_search = initialize_system()

    while True:
        try:
            print("\n You:", end=" ")
            query = input().strip()

            if query.lower() in ['quit', 'exit']:
                print("\n Thank you for using the Nursing Information Assistant. Goodbye!")
                break

            if query.lower() == 'clear':
                print_welcome()
                continue

            if not query:
                continue

            start_time = time.time()
            print("\n Assistant: ", end="")

            summary = rag_search.search_and_summarize(query, top_k=3)

            # handle dict-like output
            if isinstance(summary, dict) and "answer" in summary:
                summary = summary["answer"]

            cleaned_summary = clean_model_output(summary)
            print(cleaned_summary)

            response_time = time.time() - start_time
            print(f"\n Response time: {response_time:.2f} seconds")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {str(e)}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()
