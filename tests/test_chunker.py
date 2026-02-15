# test_chunker.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import StructureAwareChunker
parser = DocumentParser()
chunker = StructureAwareChunker(max_tokens=512, overlap_tokens=50)
doc = parser.parse("data/sample_docs/bill-20231231.pdf", "test-001")
chunks = chunker.chunk_document(doc)


print(f"Total chunks: {len(chunks)}")
print(f"Avg tokens: {sum(c.token_count for c in chunks) / len(chunks):.0f}")
print(f"Tables: {sum(1 for c in chunks if c.chunk_type == 'table')}")
for c in chunks[:15]:
    print(f"[{c.chunk_type}] ({c.token_count} tok) Section: {c.section_title}")
    print(f"  {c.text[:150]}...")
