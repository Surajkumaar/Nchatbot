# -*- coding: utf-8 -*-
"""
RAG bootstrapper: load (PDF/TXT/CSV/XLSX/DOCX/JSON/JSONL) -> chunk -> embed -> ChromaDB
- JSON auto-handling for: [{"page": "...", "content": "..."}]
- Attaches 'source_url' metadata from 'page'
- Saves to ./chromaDB and reloads
- Simple CLI for ad-hoc retrieval tests
"""

from pathlib import Path
from typing import List, Any
import traceback
import sys
import os

# ---------------- LangChain / loaders / vectorstore ----------------
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain.schema import Document
else:
    # LangChain may not be installed in the environment used to run this project.
    # Use a permissive alias so the rest of the code can type-hint Documents without a hard dependency.
    Document = Any
from src.vectorstore import ChromaVectorStore
try:
    from langchain.embeddings import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

# ---------------- Pretty banner ----------------
BANNER = r"""
============================================================
🏥 Welcome to the Nursing Information Assistant 🏥
============================================================

I can help you with information about nursing colleges and related topics.

Type 'quit' or 'exit' to end the conversation.
Type 'clear' to clear the screen.
------------------------------------------------------------
"""

# ---------------- Logging helpers ----------------
def _debug(msg: str):
    print(f"[DEBUG] {msg}")

def _info(msg: str):
    print(f"[INFO] {msg}")

def _error(msg: str, exc: Exception | None = None):
    if exc:
        print(f"[ERROR] {msg}: {exc}")
        print("[TRACE]", "".join(traceback.format_exception_only(type(exc), exc)).strip())
    else:
        print(f"[ERROR] {msg}")

# ---------------- JSON metadata helper ----------------
def _json_metadata_with_page(rec: dict, meta: dict) -> dict:
    """
    Add page URL from matched JSON record into Document metadata.
    - rec: the jq-matched object (e.g., {"page": "...", "content": "..."})
    - meta: existing metadata (contains 'source' file path)
    """
    meta = dict(meta) if meta else {}
    if isinstance(rec, dict) and "page" in rec:
        meta["page"] = rec.get("page")
        meta["source_url"] = rec.get("page")
    return meta

# ---------------- JSON/JSONL robust loader ----------------
def _load_json_like_file(path: Path) -> List[Document]:
    """
    Try to load JSON/JSONL using:
      1) Your exact structure: [{"page": "...", "content": "..."}] via '.[].content'
      2) Fallback schemas: .[], ., .items[], .documents[] (stringified)
      3) Final: full-root raw text
    """
    docs: List[Document] = []
    is_jsonl = path.suffix.lower() == ".jsonl"

    _debug(f"Loading JSON: {path}")

    # Primary: your structure
    try:
        loader = JSONLoader(
            file_path=str(path),
            jq_schema=".[].content",
            text_content=False,
            json_lines=is_jsonl,
            metadata_func=_json_metadata_with_page,
        )
        d = loader.load()
        if d:
            _debug(f"Loaded {len(d)} JSON docs ('.[].content') from {path}")
            return d
    except Exception as e:
        _debug(f"Primary schema .[].content failed for {path}: {e}")

    # Fallbacks
    fallback_schemas = (".[]", ".", ".items[]", ".documents[]")
    for schema in fallback_schemas:
        try:
            loader2 = JSONLoader(
                file_path=str(path),
                jq_schema=schema,
                text_content=True,  # stringify arbitrary JSON
                json_lines=is_jsonl,
                metadata_func=_json_metadata_with_page,
            )
            d2 = loader2.load()
            if d2:
                _debug(f"Loaded {len(d2)} JSON docs via fallback schema '{schema}' from {path}")
                return d2
        except Exception as e2:
            _debug(f"Fallback schema '{schema}' failed for {path}: {e2}")

    # Final fallback – full-root text
    try:
        loader3 = JSONLoader(
            file_path=str(path),
            jq_schema=".",
            text_content=True,
            json_lines=is_jsonl,
            metadata_func=_json_metadata_with_page,
        )
        d3 = loader3.load()
        if d3:
            _debug(f"Loaded {len(d3)} JSON docs via full-root raw text from {path}")
            return d3
    except Exception as e3:
        _error(f"Full-root raw text fallback failed for {path}", e3)

    _error(f"No JSON content extracted for {path} (file skipped)")
    return docs

# ---------------- Master loader ----------------
def load_all_documents(data_dir: str) -> List[Document]:
    """
    Recursively load all supported files from data_dir into LangChain Documents.
    Supported: PDF, TXT, CSV, Excel(xlsx), Word(docx), JSON, JSONL
    """
    data_path = Path(data_dir).resolve()
    _debug(f"Data path: {data_path}")
    documents: List[Document] = []

    # PDFs
    pdf_files = list(data_path.glob("**/*.pdf"))
    _debug(f"Found {len(pdf_files)} PDF files")
    for f in pdf_files:
        try:
            loaded = PyPDFLoader(str(f)).load()
            _debug(f"PDF: {f.name} -> {len(loaded)} docs")
            documents.extend(loaded)
        except Exception as e:
            _error(f"Failed to load PDF {f}", e)

    # TXTs
    txt_files = list(data_path.glob("**/*.txt"))
    _debug(f"Found {len(txt_files)} TXT files")
    for f in txt_files:
        try:
            loaded = TextLoader(str(f), encoding="utf-8").load()
            _debug(f"TXT: {f.name} -> {len(loaded)} docs")
            documents.extend(loaded)
        except Exception as e:
            _error(f"Failed to load TXT {f}", e)

    # CSVs
    csv_files = list(data_path.glob("**/*.csv"))
    _debug(f"Found {len(csv_files)} CSV files")
    for f in csv_files:
        try:
            loaded = CSVLoader(str(f)).load()
            _debug(f"CSV: {f.name} -> {len(loaded)} docs")
            documents.extend(loaded)
        except Exception as e:
            _error(f"Failed to load CSV {f}", e)

    # Excels
    xlsx_files = list(data_path.glob("**/*.xlsx"))
    _debug(f"Found {len(xlsx_files)} Excel files")
    for f in xlsx_files:
        try:
            loaded = UnstructuredExcelLoader(str(f)).load()
            _debug(f"XLSX: {f.name} -> {len(loaded)} docs")
            documents.extend(loaded)
        except Exception as e:
            _error(f"Failed to load Excel {f}", e)

    # Word
    docx_files = list(data_path.glob("**/*.docx"))
    _debug(f"Found {len(docx_files)} Word files")
    for f in docx_files:
        try:
            loaded = Docx2txtLoader(str(f)).load()
            _debug(f"DOCX: {f.name} -> {len(loaded)} docs")
            documents.extend(loaded)
        except Exception as e:
            _error(f"Failed to load Word {f}", e)

    # JSON & JSONL
    json_like_files = list(data_path.glob("**/*.json")) + list(data_path.glob("**/*.jsonl"))
    _debug(f"Found {len(json_like_files)} JSON/JSONL files")
    for f in json_like_files:
        try:
            loaded = _load_json_like_file(f)
            _debug(f"JSON: {f.name} -> {len(loaded)} docs")
            documents.extend(loaded)
        except Exception as e:
            _error(f"Failed to load JSON {f}", e)

    _debug(f"Total loaded documents: {len(documents)}")
    return documents

# ---------------- Chunking ----------------
def chunk_documents(documents: List[Document],
                    chunk_size: int = 1000,
                    chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)

# ---------------- Embeddings ----------------
def get_embeddings(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    _info(f"Loaded embedding model: {model_name}")
    if HuggingFaceEmbeddings is not None:
        return HuggingFaceEmbeddings(model_name=model_name)
    # Fallback: return a SentenceTransformer instance if LangChain's wrapper isn't available
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        _error("No embedding implementation available (install langchain or sentence-transformers)")
        return None

# ---------------- Vector store helpers ----------------
def build_and_save_chroma(docs: List[Document],
                          embeddings,
                          chroma_dir: str = "chromaDB",
                          collection_name: str = "nursing_collection") -> ChromaVectorStore:
    _info(f"Building vector store from {len(docs)} raw documents...")
    store = ChromaVectorStore(persist_dir=chroma_dir)
    store.build_from_documents(docs)
    _info(f"Vector store built and saved to {chroma_dir}")
    return store


def load_chroma(chroma_dir: str = "chromaDB",
                embeddings=None,
                collection_name: str = "nursing_collection") -> ChromaVectorStore:
    _info("Loaded Chroma collection from " + chroma_dir)
    store = ChromaVectorStore(persist_dir=chroma_dir)
    store.load()
    return store

# ---------------- Simple CLI for testing ----------------
def interactive_cli(vdb, k: int = 5):
    print("\nYou can now query the vector store. Type your question below.")
    while True:
        try:
            q = input("\nAsk: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            print("Bye!")
            break
        if q.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print(BANNER)
            continue

        hits = vdb.similarity_search(q, k=k)
        print(f"\nTop {k} matches:")
        for i, h in enumerate(hits, 1):
            src_path = h.metadata.get("source")
            src_url = h.metadata.get("source_url")
            print(f"\n[{i}] score≈cos not shown (Chroma-only)")
            if src_url:
                print(f"    Source URL : {src_url}")
            if src_path:
                print(f"    Source File: {src_path}")
            preview = (h.page_content or "").replace("\n", " ")
            print("    Text:", preview[:400], "..." if len(preview) > 400 else "")

# ---------------- Main ----------------
if __name__ == "__main__":
    print(BANNER)
    print("🔄 Initializing the system...")

    # Point to your BASE folder (not just pdf/)
    # Example: r"C:\Users\STIC-11\Desktop\Nchat\rag1"
    base_path = r"C:\Users\hamsa\OneDrive\Desktop\my proj\Nursing chatbot\NCH_gemini\Nchatbot"

    # 1) Load
    raw_docs = load_all_documents(base_path)

    # 2) Chunk
    _info(f"Split {len(raw_docs)} documents into chunks.")
    chunks = chunk_documents(raw_docs, chunk_size=1000, chunk_overlap=150)
    _info(f"Chunks created: {len(chunks)}")

    # 3) Embeddings
    embed_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = get_embeddings(embed_model_name)

    # 4) Build / Save ChromaDB
    vectordb = build_and_save_chroma(chunks, embeddings, chroma_dir="chromaDB")

    # 5) Reload Chroma (demonstrate load path)
    vectordb = load_chroma("chromaDB", embeddings=embeddings)

    # 6) (Optional) Show a couple of examples after load
    shown = 0
    for d in raw_docs:
        if isinstance(d, Document) and (d.metadata.get("source_url") or d.metadata.get("source")):
            src = d.metadata.get("source_url") or d.metadata.get("source")
            print(f"Example source: {src}")
            print("Preview:", (d.page_content or "")[:200].replace("\n", " "), "...\n")
            shown += 1
            if shown >= 2:
                break

    # 7) Interactive CLI
    try:
        # Simulate model loading banner (if you later integrate a local LLM)
        print("🔹 Loading Sarvam-2B model...")
        print("✅ Sarvam-2B model loaded successfully! (placeholder banner)")
    except Exception:
        pass

    interactive_cli(vectordb, k=5)
