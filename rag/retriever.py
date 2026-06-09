import os, pickle
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_retriever(index_path="data/faiss_index"):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local(index_path, embeddings,
                            allow_dangerous_deserialization=True)

def retrieve(question: str, k=5) -> list:
    vs = get_retriever()
    return vs.similarity_search(question, k=k)
