from dataclasses import dataclass
from app.ingestion.parser import DocumentBlock, ParsedDocument, BlockType
import re
from app.ingestion.preprocessor import TextPreprocessor

@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    page_number: int
    section_title: str = ""
    parent_section: str = ""
    chunk_type: str = "paragraph"
    token_count: int = 0


    def to_dict(self) -> dict:
        return {
            "text": self.text, "doc_id": self.doc_id,
            "chunk_index": self.chunk_index, "page_number": self.page_number,
            "section_title": self.section_title, "parent_section": self.parent_section,
            "chunk_type": self.chunk_type, "token_count": self.token_count,
        }


class StructureAwareChunker:
    def __init__(self, max_tokens=512, overlap_tokens=50):
        self.max_tokens = max_tokens
        self.min_chunk_tokens = 50
        self.overlap_tokens = overlap_tokens
        self.preprocessor = TextPreprocessor()


    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        chunks = []
        chunk_idx = 0
        sections = self._group_by_section(doc.blocks)


        for section_title, blocks in sections:
            for block in blocks:
                if block.block_type == BlockType.TABLE:
                    # Tables always stay as single chunks
                    chunks.append(self._make_chunk(block, doc.doc_id, chunk_idx))
                    chunk_idx += 1
                elif self.preprocessor.count_tokens(block.content) <= self.max_tokens:
                    chunks.append(self._make_chunk(block, doc.doc_id, chunk_idx))
                    chunk_idx += 1
                else:
                    # Split large blocks at sentence boundaries
                    for sub in self._split_with_overlap(block, doc.doc_id):
                        sub.chunk_index = chunk_idx
                        chunks.append(sub)
                        chunk_idx += 1


        chunks = self._merge_small_chunks(chunks)
        return chunks
    
    def _group_by_section(self, blocks: list[DocumentBlock]) -> list[tuple[str, list[DocumentBlock]]]:
        sections = []
        current_section = ""
        current_blocks = []


        for block in blocks:
            if block.block_type == BlockType.HEADING:
                if current_blocks:
                    sections.append((current_section, current_blocks))
                current_section = block.content.strip()
                current_blocks = [block]
            else:
                current_blocks.append(block)


        if current_blocks:
            sections.append((current_section, current_blocks))
        return sections
    
    def _make_chunk(self, block: DocumentBlock, doc_id: str, chunk_idx: int) -> Chunk:
        return Chunk(
            text=block.content,
            doc_id=doc_id,
            chunk_index=chunk_idx,
            page_number=block.page_number,
            section_title=block.section_title,
            parent_section=block.parent_section,
            chunk_type=block.block_type.value,
            token_count=self.preprocessor.count_tokens(block.content)
        )
    
    def _split_with_overlap(self, block: DocumentBlock, doc_id: str) -> list[Chunk]:
        sentences = re.split(r'(?<=[.!?]) +', block.content)
        chunks = []
        current_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.preprocessor.count_tokens(sentence)

            if current_tokens + sentence_tokens > self.max_tokens and current_sentences:
                # Save current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text.strip(),
                    doc_id=doc_id,
                    chunk_index=-1,
                    page_number=block.page_number,
                    section_title=block.section_title,
                    parent_section=block.parent_section,
                    chunk_type=block.block_type.value,
                    token_count=self.preprocessor.count_tokens(chunk_text)
                ))

                # Build overlap from previous sentences
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = self.preprocessor.count_tokens(s)
                    if overlap_tokens + s_tokens > self.overlap_tokens:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text.strip(),
                doc_id=doc_id,
                chunk_index=-1,
                page_number=block.page_number,
                section_title=block.section_title,
                parent_section=block.parent_section,
                chunk_type=block.block_type.value,
                token_count=self.preprocessor.count_tokens(chunk_text)
            ))

        return chunks
    
    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks or len(chunks) == 1:
            return chunks
        
        merged_chunks = []
        curr_buffer = [chunks[0]]
        curr_tokens = chunks[0].token_count
        curr_chunk_type = chunks[0].chunk_type
        curr_section = chunks[0].section_title

        for chunk in chunks[1:]:
            curr_chunk_tokens = chunk.token_count
            if (chunk.chunk_type != curr_chunk_type 
                or chunk.section_title != curr_section 
                or curr_tokens + curr_chunk_tokens > self.max_tokens
                or (curr_tokens >= self.min_chunk_tokens and curr_chunk_tokens >= self.min_chunk_tokens)):
                # create merged chunk
                merged_text = " ".join(c.text for c in curr_buffer)
                merged_tokens = sum(c.token_count for c in curr_buffer)
                new_chunk = curr_buffer[0]
                new_chunk.text = merged_text
                new_chunk.token_count = merged_tokens
                merged_chunks.append(new_chunk)
                curr_buffer = [chunk]
                curr_tokens = curr_chunk_tokens
                curr_chunk_type = chunk.chunk_type
                curr_section = chunk.section_title
            else:
                curr_buffer.append(chunk)
                curr_tokens += curr_chunk_tokens
        
        if curr_buffer:
            merged_text = " ".join(c.text for c in curr_buffer)
            merged_tokens = sum(c.token_count for c in curr_buffer)
            new_chunk = curr_buffer[0]
            new_chunk.text = merged_text
            new_chunk.token_count = merged_tokens
            merged_chunks.append(new_chunk)

        # Reindex the merged chunks
        for idx, chunk in enumerate(merged_chunks):
            chunk.chunk_index = idx

        return merged_chunks
                
