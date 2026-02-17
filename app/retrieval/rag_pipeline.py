from openai import OpenAI
from dataclasses import dataclass

from app.config import settings
from app.retrieval.vector_store import VectorStore

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    query: str


class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = OpenAI(api_key=settings.openai_api_key)


    def query(self, question: str, doc_filter=None, top_k=5) -> RAGResponse:
        # Step 1: Retrieve relevant chunks
        results = self.vector_store.search(question, top_k, doc_filter)
        if not results:
            return RAGResponse(answer="No relevant info found.", sources=[], query=question)


        # Step 2: Build context with source labels
        context_parts = []
        for i, r in enumerate(results):
            label = f"[Source {i+1}] Section: {r['section_title']} (Page {r['page_number']})"
            context_parts.append(f"{label}\n{r['text']}")
        context = "\n\n---\n\n".join(context_parts)


        # Step 3: Generate answer
        system = """Answer based on the provided context only.
If the answer isn't in the context, say so clearly.
Cite sources using [Source N] references. Be concise but thorough."""


        response = self.llm.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.1, max_tokens=1000
        )

        return RAGResponse(
            answer=response.choices[0].message.content,
            sources=results, query=question
        )
