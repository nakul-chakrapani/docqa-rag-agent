from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
import uuid
from app.config import settings
from app.retrieval.embedder import EmbeddingService
from app.ingestion.chunker import Chunk

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection = settings.collection_name
        self.embedder = EmbeddingService()
        self._ensure_collection()


    def _ensure_collection(self):
        distance_metric = Distance.COSINE
        vector_dimension = settings.embedding_dimension
        collection_name = settings.collection_name

        # Create collection with cosine distance if not exists
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dimension, distance=distance_metric)
            )

            # Create payload indexes on: doc_id, section_title, chunk_type
            payload_indices = ["doc_id", "section_title", "chunk_type"]
            for field in payload_indices:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema={"type": "keyword"}
                )



    def add_chunks(self, chunks: list[Chunk]) -> int:
        # 1. Embed all chunk texts in batch
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)

        # 2. Create PointStruct for each (uuid, vector, payload=chunk.to_dict())
        points_to_upsert = []
        for chunk, vector in zip(chunks, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=chunk.to_dict()
            )
            points_to_upsert.append(point)
        
        # 3. Upsert in batches of 100
        for i in range(0, len(points_to_upsert), 100):
            batch = points_to_upsert[i:i+100]
            self.client.upsert(collection_name=self.collection, points=batch)

        # Return count of points added
        return len(points_to_upsert)


    def search(self, query: str, top_k=20, doc_filter=None) -> list[dict]:
        # 1. Embed the query
        query_vector = self.embedder.embed_text(query)

        # 2. Build optional Filter if doc_filter provided
        if doc_filter:
            filter_conditions = []
            for key, value in doc_filter.items():
                condition = FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                filter_conditions.append(condition)
            qdrant_filter = Filter(must=filter_conditions)
        else:
            qdrant_filter = None

        # 3. Search collection with query_vector, limit, filter
        search_results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter
        )

        # 4. Return list of {text, score, doc_id, page_number, section_title}
        results = []
        for res in search_results:
            payload = res.payload
            results.append({
                "text": payload.get("text", ""),
                "score": res.score,
                "doc_id": payload.get("doc_id", ""),
                "page_number": payload.get("page_number", -1),
                "section_title": payload.get("section_title", "")
            })
        return results


    def delete_document(self, doc_id: str):
        # Delete all points where payload doc_id matches
        self.client.delete(
            collection_name=self.collection,
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        )
