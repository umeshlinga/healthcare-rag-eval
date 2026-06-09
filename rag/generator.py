import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

PROMPT_VERSION = "v2.1"

SYSTEM = """You are a Medicare/Medicaid benefits assistant.
Answer ONLY using the provided context.
If the context does not contain enough information, say exactly:
I don't have enough information in my knowledge base to answer this accurately.
Always cite the source document. Never guess beyond what is stated."""

HUMAN = """Context:
{context}

Member question: {question}

Answer (cite sources):"""

def generate_answer(question: str, context_docs: list) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                     openai_api_key=os.getenv("OPENAI_API_KEY"))
    context = "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('source_file','unknown')}, Page {d.metadata.get('page','?')}]\n{d.page_content}"
        for d in context_docs
    ])
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
    result = (prompt | llm).invoke({"context": context, "question": question})
    return {"answer": result.content, "context": context,
            "sources": list(set(d.metadata.get("source_file") for d in context_docs)),
            "prompt_version": PROMPT_VERSION}
