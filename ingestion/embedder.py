import os, pickle
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def build_index(chunks_path="data/processed/chunks.pkl",
                save_path="data/faiss_index"):
    print("Loading chunks...")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Embedding {len(chunks)} chunks with text-embedding-3-small...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"FAISS index saved to {save_path}")
    print(f"Index contains {vectorstore.index.ntotal} vectors")
    return vectorstore

if __name__ == "__main__":
    build_index()
