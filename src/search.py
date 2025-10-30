import os
import torch
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFacePipeline
from deep_translator import GoogleTranslator
from langdetect import detect
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", 
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        # Initialize vectorstore
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # Initialize Sarvam-2B model
        print("🔹 Loading Sarvam-2B model...")
        model_name = "sarvamai/sarvam-2b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Create generation pipeline
        gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
        )
        self.llm = HuggingFacePipeline(pipeline=gen_pipeline)
        print("✅ Sarvam-2B model loaded successfully!")

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except:
            return 'en'

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
            
        # Create prompt for the model
        prompt = f"""Based on the following context, answer the query: '{query}'\n\nContext:\n{context}\n\nAnswer:"""
        
        # Get response from Sarvam-2B
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
