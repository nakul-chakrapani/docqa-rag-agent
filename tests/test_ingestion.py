import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import StructureAwareChunker
from app.retrieval.vector_store import VectorStore



# test_ingestion.py
parser = DocumentParser()
chunker = StructureAwareChunker()
vector_store = VectorStore()


docs = [
    ("data/sample_docs/bill-20231231.pdf", "bill-20231231"),
    ("data/sample_docs/bill-20240331.pdf", "bill-20240331"),
    ("data/sample_docs/bill-20240930.pdf", "bill-20240930"),
    ("data/sample_docs/bill-20241231.pdf", "bill-20241231"),
]

for file_path, doc_id in docs:
    doc = parser.parse(file_path, doc_id)
    chunks = chunker.chunk_document(doc)
    count = vector_store.add_chunks(chunks)
    print(f"Stored {count} chunks")


results = vector_store.search("What is the Total Revenue in 2024 first quarter?")
for r in results[:3]:
    print(f"Score: {r['score']:.4f} | Section: {r['section_title']} | {r['text'][:100]}")
