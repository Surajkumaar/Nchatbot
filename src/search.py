import os
from dotenv import load_dotenv
from src.vectorstore import ChromaVectorStore
from src.data_loader import load_all_documents
from sentence_transformers import SentenceTransformer
from langdetect import detect
import requests
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
load_dotenv()


class GeminiLLM:
    """Gemini API wrapper with multilingual support and generation parameters"""
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = None,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 512
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        # allow override via env var; default to gemini-2.0-flash (from your Postman example)
        self.model = model or os.getenv("GOOGLE_GEMINI_MODEL") or "gemini-2.0-flash"
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        # Use the generateContent endpoint (v1beta) compatible with Gemini models
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        if not self.api_key:
            print("[WARNING] No GOOGLE_API_KEY found. Set it in .env or pass directly.")
        else:
            print(f"[INFO] Gemini LLM initialized: {self.model}")
            print(f"[INFO] Generation config - Temperature: {self.temperature}, Top-P: {self.top_p}, Top-K: {self.top_k}")
    
    def invoke(self, prompt: str) -> str:
        """Invoke Gemini API with generation parameters"""
        if not self.api_key:
            return "Error: No GOOGLE_API_KEY configured. Set GOOGLE_API_KEY in your environment or .env to use the Gemini API."
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        # Use the correct v1beta generateContent body format (matching your Postman request)
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "topK": self.top_k,
                "maxOutputTokens": self.max_output_tokens
            }
        }

        try:
            resp = requests.post(self.endpoint, json=body, headers=headers, timeout=60)
            if resp.status_code == 404:
                return (f"Error: 404 - model '{self.model}' not found or not enabled for this API. "
                        "Set GOOGLE_GEMINI_MODEL to a model your API key can access (for example 'models/text-bison-001'), or enable the model for your project.")
            if resp.status_code == 403:
                return f"Error: 403 - access denied for model '{self.model}'. Check permissions/quotas."
            resp.raise_for_status()
            data = resp.json()
            # handle possible response shapes
            if isinstance(data, dict):
                if "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"])>0:
                    return data["candidates"][0].get("content", {}).get("parts", [])[0].get("text", "").strip()
                if "outputs" in data and isinstance(data["outputs"], list) and len(data["outputs"])>0:
                    out = data["outputs"][0]
                    if isinstance(out, dict) and "content" in out:
                        c = out["content"]
                        if isinstance(c, list) and len(c)>0 and isinstance(c[0], dict) and "text" in c[0]:
                            return c[0]["text"].strip()
                        if isinstance(c, str):
                            return c.strip()
                if "output" in data and isinstance(data["output"], str):
                    return data["output"].strip()
            return str(data)
        except Exception as e:
            return f"Error: {e}"


class RAGSearch:
    """
    Multilingual RAG system for nursing queries
    Supports: Tamil, Hindi, Malayalam, English
    """
    
    # Language mapping for better detection
    SUPPORTED_LANGUAGES = {
        'ta': 'Tamil',
        'hi': 'Hindi',
        'ml': 'Malayalam',
        'en': 'English'
    }
    
    def __init__(
        self, 
        persist_dir: str = "chromaDB", 
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        data_path: str = "pdf",
        temperature: float = 0.4,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 1024
    ):
        print("[INFO] Initializing Multilingual RAG Search System...")
        print(f"[INFO] Supported languages: {', '.join(self.SUPPORTED_LANGUAGES.values())}")
        
        # Initialize vector store using ChromaVectorStore from vectorstore.py
        # Using multilingual embedding model for cross-language support
        self.vectorstore = ChromaVectorStore(
            persist_dir=persist_dir, 
            embedding_model=embedding_model
        )
        
       

        # Load or build vector store
        try:
            self.vectorstore.load()
            count = self.vectorstore.collection.count() if hasattr(self.vectorstore.collection, "count") else 0
            
            if count == 0:
                print("[INFO] Empty vector store detected. Building from documents...")
                docs = load_all_documents(data_path)
                self.vectorstore.build_from_documents(docs)
            else:
                print(f"[INFO] Loaded existing vector store with {count} documents")
                
        except Exception as e:
            print(f"[INFO] Building new vector store: {str(e)}")
            docs = load_all_documents(data_path)
            self.vectorstore.build_from_documents(docs)
        
        # Initialize Gemini LLM with parameters
        # Gemini natively supports Tamil, Hindi, Malayalam, and English
        self.llm = GeminiLLM(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens
        )
        
        # Option 2: Direct API key (uncomment to use)
        # self.llm = GeminiLLM(
        #     api_key="YOUR_API_KEY_HERE",
        #     temperature=temperature,
        #     top_p=top_p,
        #     top_k=top_k,
        #     max_output_tokens=max_tokens
        # )
        
        print("[INFO] Multilingual RAG Search initialized successfully\n")

    def detect_language(self, text: str) -> str:
        """
        Detect language of input text
        Returns: Language code (ta, hi, ml, en)
        """
        try:
            lang_code = detect(text)
            
            # Map detected language to supported languages
            if lang_code in self.SUPPORTED_LANGUAGES:
                print(f"[INFO] Detected language: {self.SUPPORTED_LANGUAGES[lang_code]} ({lang_code})")
                return lang_code
            else:
                print(f"[INFO] Language {lang_code} not in supported list, defaulting to English")
                return 'en'
        except Exception as e:
            print(f"[WARNING] Language detection failed: {e}. Defaulting to English")
            return 'en'

    def get_language_instruction(self, lang_code: str) -> str:
        """Get language-specific instruction for Gemini"""
        language_instructions = {
            'ta': 'தமிழில் பதிலளிக்கவும் (Answer in Tamil)',
            'hi': 'हिंदी में उत्तर दें (Answer in Hindi)',
            'ml': 'മലയാളത്തിൽ ഉത്തരം നൽകുക (Answer in Malayalam)',
            'en': 'Answer in English'
        }
        return language_instructions.get(lang_code, 'Answer in English')

    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        """
        Multilingual search and response generation
        
        Args:
            query: User question (can be in Tamil, Hindi, Malayalam, or English)
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Answer in the same language as the query
        """
        # Detect query language
        query_language = self.detect_language(query)
        language_name = self.SUPPORTED_LANGUAGES.get(query_language, 'English')
        
        print(f"[INFO] Processing query in {language_name}")
        print(f"[INFO] Query: {query}")
        
        # Query vector store (multilingual embeddings handle cross-language search)
        print(f"[INFO] Searching vector store...")
        results = self.vectorstore.query(query, top_k=top_k)
        
        # Extract document text from results
        texts = [r["document"] for r in results if r.get("document")]
        
        if not texts:
            # Return "no results" message in user's language
            no_results_messages = {
                'ta': 'மன்னிக்கவும், உங்கள் கேள்விக்கு தொடர்புடைய தகவல் கிடைக்கவில்லை. செவிலியர் கல்வி தொடர்பான கேள்விகளை கேளுங்கள்.',
                'hi': 'क्षमा करें, आपके प्रश्न के लिए प्रासंगिक जानकारी नहीं मिली। कृपया नर्सिंग शिक्षा से संबंधित प्रश्न पूछें।',
                'ml': 'ക്ഷമിക്കണം, നിങ്ങളുടെ ചോദ്യത്തിന് പ്രസക്തമായ വിവരങ്ങൾ കണ്ടെത്താനായില്ല. ദയവായി നഴ്‌സിംഗ് വിദ്യാഭ്യാസവുമായി ബന്ധപ്പെട്ട ചോദ്യങ്ങൾ ചോദിക്കുക.',
                'en': 'I apologize, but I couldn\'t find relevant nursing-related information for your query. Please ask questions about nursing education, curriculum, or related topics.'
            }
            return no_results_messages.get(query_language, no_results_messages['en'])
        
        # Build context from retrieved documents
        context = "\n\n".join(texts)
        
        # Get language-specific instruction
        language_instruction = self.get_language_instruction(query_language)
        
        # Create multilingual nursing-specific prompt
        prompt = f"""You are a specialized nursing education assistant for Tamil Nadu government. You support Tamil, Hindi, Malayalam, and English languages.

STRICT RULES:
1. Answer ONLY nursing-related questions (nursing education, curriculum, colleges, healthcare topics)
2. If the question is NOT about nursing or healthcare, politely decline in the user's language
3. Base your answer ONLY on the provided context
4. Be concise, accurate, and culturally appropriate
5. IMPORTANT: {language_instruction}

Context from nursing documents:
{context}

User Question ({language_name}): {query}

Answer (in {language_name}, nursing-related information only):"""
        
        # Get response from Gemini (natively supports multilingual output)
        print(f"[INFO] Generating response in {language_name}...")
        response = self.llm.invoke(prompt)
        
        return response


# Example usage and testing
if __name__ == "__main__":
    # Initialize multilingual RAG system
    rag = RAGSearch(
        persist_dir="chromaDB",
        data_path="pdf",
        temperature=0.4,     # Lower for more focused answers
        top_p=0.95,          # Nucleus sampling
        top_k=40,            # Top-k sampling
        max_tokens=1024      # Max response length
    )
    
    # Test queries in multiple languages
    test_queries = [
        # English
        "What is the nursing curriculum for first year?",
        
        # Tamil
        "முதல் ஆண்டு செவிலியர் பாடத்திட்டம் என்ன?",
        
        # Hindi
        "पहले साल का नर्सिंग पाठ्यक्रम क्या है?",
        
        # Malayalam
        "ഒന്നാം വർഷത്തെ നഴ്സിംഗ് പാഠ്യപദ്ധതി എന്താണ്?",
        
        # Non-nursing query (to test restriction)
        "What is machine learning?"
    ]
    
    print("\n" + "="*80)
    print("TESTING MULTILINGUAL RAG SEARCH SYSTEM")
    print("="*80 + "\n")
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print(f"{'─'*80}\n")
        
        answer = rag.search_and_summarize(query, top_k=3)
        print(f"Answer:\n{answer}\n")
