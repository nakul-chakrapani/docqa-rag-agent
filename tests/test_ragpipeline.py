import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.retrieval.rag_pipeline import RAGPipeline

rag = RAGPipeline()
questions = [
    # "What was the total revenue reported?",           # Factual
    # "Summarize the key findings.",                    # Summary
    # "What are the main risk factors?",                # Extraction
    # "Tell me about something not in the document.",   # Fallback test
    # "What is the name of my dog?"
    "What is the Total Revenue of BILL Holdings in 2024?",
    "Who is the CEO of BILL Holdings?",
    "BILL's float revenue",

]
for q in questions:
    resp = rag.query(q)
    print(f"Q: {q}")
    print(f"A: {resp.answer}")
    print(f"Sources: {len(resp.sources)}\n")
