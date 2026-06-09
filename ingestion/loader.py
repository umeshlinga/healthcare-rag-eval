import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def classify_doc(fname: str) -> str:
    fname = fname.lower()
    if "cost" in fname or "price" in fname: return "cost_sharing"
    if "preventive" in fname: return "preventive_care"
    if "prior" in fname: return "prior_auth"
    if "benefit" in fname: return "benefits"
    return "general_coverage"

def load_and_chunk(pdf_dir: str, chunk_size=512, overlap=64):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    all_chunks = []
    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue
        print(f"Loading: {fname}")
        loader = PyPDFLoader(os.path.join(pdf_dir, fname))
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        for i, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = fname
            chunk.metadata["doc_type"] = classify_doc(fname)
            chunk.metadata["chunk_id"] = f"{fname}_{i}"
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks from {len(pages)} pages")
    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    chunks = load_and_chunk("data/raw")
    sample = chunks[0]
    print(f"\nSample chunk:")
    print(f"  doc_type: {sample.metadata['doc_type']}")
    print(f"  source:   {sample.metadata['source_file']}")
    print(f"  text:     {sample.page_content[:200]}")
