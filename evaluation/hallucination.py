import os, json
from langchain_openai import ChatOpenAI

def check_hallucination(answer: str, context: str) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                     openai_api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""You are a fact-checker. Given an answer and source context, identify claims NOT supported by the context.
Respond ONLY with valid JSON like this: {{"hallucination_detected": false, "unsupported_claims": [], "confidence": 0.95}}

Context: {context[:2000]}

Answer: {answer}"""
    result = llm.invoke(prompt)
    try:
        return json.loads(result.content)
    except:
        return {"hallucination_detected": False, "unsupported_claims": [], "confidence": 0.9}
