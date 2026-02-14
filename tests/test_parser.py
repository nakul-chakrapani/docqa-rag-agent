import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.parser import PDFParser, BlockType
parser = PDFParser()
doc = parser.parse("data/sample_docs/bill-20231231.pdf", "test-001")
print(f"Pages: {doc.total_pages}, Blocks: {len(doc.blocks)}")
print(f"Headings: {sum(1 for b in doc.blocks if b.block_type == BlockType.HEADING)}")
print(f"Tables: {sum(1 for b in doc.blocks if b.block_type == BlockType.TABLE)}")
for block in doc.blocks:
    # Print blocks from specific page
    if block.page_number == 10:
        print(f"[{block.block_type.value}] p.{block.page_number}: {block.content[:80]}...")
