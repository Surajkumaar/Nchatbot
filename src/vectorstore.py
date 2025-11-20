import os
import uuid
import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    """Simple ChromaDB-backed vector store wrapper.

    - persist_dir: root directory where Chroma will persist its DB files
    - embedding_model: sentence-transformers model used for query-time encoding
    """
    def __init__(self, persist_dir: str = "chromaDB", embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        # initialize chroma client and a collection
        self.client = chromadb.Client(Settings(persist_directory=self.persist_dir))
        self.collection_name = "nursing_collection"
        # create or get existing collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name)

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building Chroma vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        # If no chunks were produced, nothing to add
        if not chunks:
            print("[WARN] No document chunks produced; skipping vector store build.")
            return
        embeddings = emb_pipe.embed_chunks(chunks)
        # diagnostic logging
        try:
            print(f"[DEBUG] Number of chunks: {len(chunks)}; example chunk text length: {len(chunks[0].page_content) if len(chunks)>0 else 0}")
        except Exception:
            pass
        texts = [chunk.page_content for chunk in chunks]
        # Ensure embeddings is a numpy array
        try:
            embeddings = np.array(embeddings)
        except Exception:
            embeddings = None

        # If embeddings are empty, attempt a fallback using the local SentenceTransformer model
        if embeddings is None or (hasattr(embeddings, 'size') and embeddings.size == 0):
            print("[WARN] Embeddings are empty from EmbeddingPipeline; attempting fallback using SentenceTransformer...")
            try:
                texts = [chunk.page_content for chunk in chunks]
                fallback_emb = self.model.encode(texts, show_progress_bar=False)
                embeddings = np.array(fallback_emb)
                print(f"[INFO] Fallback embeddings shape: {embeddings.shape}")
            except Exception as e:
                print(f"[ERROR] Fallback embedding computation failed: {e}")
                return
        metadatas = [{"text": t} for t in texts]
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        # chroma expects python lists (not numpy arrays)
        try:
            # final validation before sending to Chroma
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Embeddings are empty after fallback — aborting add operation.")
            # ensure embeddings is a list of vectors matching documents
            emb_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
            if len(emb_list) != len(texts):
                raise ValueError(f"Embeddings count ({len(emb_list)}) does not match documents count ({len(texts)})")
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=emb_list)
        except Exception as e:
            print(f"[ERROR] Failed to add embeddings to Chroma collection: {e}")
            return
        # persist() may not exist on all Chroma client builds; ignore if unavailable
        try:
            if hasattr(self.client, "persist"):
                self.client.persist()
        except Exception:
            pass
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None, documents: List[str] = None):
        ids = [str(uuid.uuid4()) for _ in range(embeddings.shape[0])]
        docs = documents if documents is not None else [m.get("text", "") for m in (metadatas or [])]
        metas = metadatas or [{} for _ in range(embeddings.shape[0])]
        self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings.tolist())
        try:
            if hasattr(self.client, "persist"):
                self.client.persist()
        except Exception:
            pass
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Chroma collection.")

    def save(self):
        # chroma persists automatically for many setups; try to call persist if available
        try:
            if hasattr(self.client, "persist"):
                self.client.persist()
        except Exception:
            pass
        print(f"[INFO] Persisted Chroma DB to {self.persist_dir}")

    def load(self):
        # collection is lazy-loaded by chroma client; re-create client to be safe
        self.client = chromadb.Client(Settings(persist_directory=self.persist_dir))
        self.collection = self.client.get_collection(name=self.collection_name)
        print(f"[INFO] Loaded Chroma collection '{self.collection_name}' from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        # chroma expects list-like embeddings
        resp = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k, include=["metadatas", "documents", "distances"])
        results = []
        # results structure: {"ids": [[...]], "metadatas": [[...]], "documents": [[...]], "distances": [[...]]}
        ids = resp.get("ids", [[]])[0]
        dists = resp.get("distances", [[]])[0]
        metas = resp.get("metadatas", [[]])[0]
        docs = resp.get("documents", [[]])[0]
        for idx, dist, meta, doc in zip(ids, dists, metas, docs):
            results.append({"id": idx, "distance": dist, "metadata": meta, "document": doc})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying Chroma DB for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)


# Example usage
if __name__ == "__main__":
    from src.data_loader import load_all_documents
    docs = load_all_documents("data")
    store = ChromaVectorStore("chromaDB")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
