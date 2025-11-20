import os
import requests
from dotenv import load_dotenv
from src.vectorstore import ChromaVectorStore
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
load_dotenv()



class GeminiLLM:
    """Gemini 1.5 Flash API Wrapper"""
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = None,
        temperature: float = 0.4,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 1024
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model or os.getenv("GOOGLE_GEMINI_MODEL") or "models/gemini-1.5-flash"
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens

        #  Use the correct Gemini 1.5 endpoint
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:generateContent"
        
        if not self.api_key:
            print("[WARNING] No GOOGLE_API_KEY found. Set it in .env or pass directly.")
        else:
            print(f"[INFO] Gemini LLM initialized: {self.model}")
            print(f"[INFO] Generation config - Temp: {self.temperature}, Top-P: {self.top_p}, Top-K: {self.top_k}")

    def invoke(self, prompt: str) -> str:
        """Send request to Gemini 1.5 Flash"""
        if not self.api_key:
            return "Error: No GOOGLE_API_KEY configured."
        
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}

        #  Gemini 1.5 uses `contents`, not `prompt`
        body = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "topK": self.top_k,
                "maxOutputTokens": self.max_output_tokens
            }
        }

        try:
            resp = requests.post(self.endpoint, params=params, json=body, headers=headers, timeout=60)

            # Handle common errors
            if resp.status_code == 404:
                return ("Error: 404 - Model not found. "
                        "Set GOOGLE_GEMINI_MODEL=models/gemini-1.5-flash and ensure your API key is from Google AI Studio.")
            if resp.status_code == 403:
                return "Error: 403 - Access denied. Enable Gemini 1.5 in your project or check API key."

            resp.raise_for_status()
            data = resp.json()

            #  Extract text output correctly for Gemini 1.5
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except Exception as e:
            return f"Error: {e}"


class RAGSearch:
    def __init__(self, persist_dir: str = "chromaDB", 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        # Initialize vectorstore
        self.vectorstore = ChromaVectorStore(persist_dir, embedding_model)
        self.llm = GeminiLLM(api_key=os.getenv("GOOGLE_API_KEY"))

        # Load vectorstore; if collection is empty, build from documents.
        # Rationale: the directory may exist (e.g., contains a .gitkeep) but the collection
        # could still be empty — check collection size instead of filesystem contents.
        self.vectorstore.load()
        try:
            count = 0
            if hasattr(self.vectorstore.collection, "count"):
                # chroma Collection.count() returns number of items
                count = self.vectorstore.collection.count()
            else:
                # Fallback: attempt to query for any id
                resp = self.vectorstore.collection.get(include=["ids"])
                ids = resp.get("ids", [])
                # ids may be a flat list or nested; handle both
                if isinstance(ids, list):
                    if len(ids) == 0:
                        count = 0
                    elif isinstance(ids[0], list):
                        count = len(ids[0])
                    else:
                        count = len(ids)
        except Exception:
            count = 0

        if count == 0:
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)

        # Initialize Gemini API client wrapper
        print("🔹 Configuring Gemini API client...")
        self.llm = GeminiLLM()
        print("✅ Gemini API client ready.")

        # Load system prompt once; keep in memory for reuse
        self.system_prompt = self._load_system_prompt()

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except:
            return 'en'

    def _load_system_prompt(self, path: str = None) -> str:
        """Load nursing system prompt from file. Returns a default fallback if not available."""
        if path is None:
            # relative to project src directory
            path = Path(__file__).parent / "nursing_system_prompt.txt"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            # Minimal fallback system prompt
            return (
                """
You are a Nursing Information Assistant. Use the provided context to answer administrative and educational
questions about nursing programs. Do not give medical advice. If the answer is not in the context, say you
don't know and suggest looking at official regulatory sources or asking for a broader search. Answer in user's language.
"""
            )

    def translate_if_needed(self, text: str, target_lang: str = 'en') -> str:
        source_lang = self.detect_language(text)
        if source_lang != target_lang:
            try:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                return translator.translate(text)
            except:
                return text
        return text

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        # Translate query to English if it's in another language
        query_en = self.translate_if_needed(query)
        
        # Search for relevant documents
        results = self.vectorstore.query(query_en, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        
        if not context:
            return "No relevant documents found."
            
        # Create prompt for the model. Prepend the nursing system prompt as an instruction block.
        sys_prompt = (self.system_prompt + "\n\n") if getattr(self, "system_prompt", None) else ""
        prompt = (
            f"{sys_prompt}Based on the following context, answer the query: '{query}'\n\nContext:\n{context}\n\nAnswer:"
        )
        
        # Get response from LLM (Gemini)
        response = self.llm.invoke(prompt)
        
        # Translate response back to query language if needed
        query_lang = self.detect_language(query)
        if query_lang != 'en':
            response = self.translate_if_needed(response, query_lang)
            
        return response

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
