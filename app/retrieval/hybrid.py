class HybridRetriever:
    def __init__(self, vector_store, bm25_store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf_k = 60  # Standard RRF constant


    def search(self, query, top_k=5, doc_filter=None):
        vector_results = self.vector_store.search(query, top_k=20, doc_filter=doc_filter)
        bm25_results = self.bm25_store.search(query, top_k=20, doc_filter=doc_filter)


        rrf_scores = {}  # text_key -> {score, data}
        for rank, r in enumerate(vector_results, 1):
            key = r["text"][:100]
            rrf_scores.setdefault(key, {"score": 0, "data": r})
            rrf_scores[key]["score"] += 1.0 / (self.rrf_k + rank)


        for rank, r in enumerate(bm25_results, 1):
            key = r["text"][:100]
            rrf_scores.setdefault(key, {"score": 0, "data": r})
            rrf_scores[key]["score"] += 1.0 / (self.rrf_k + rank)


        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [{"hybrid_score": item["score"], **item["data"]} for item in sorted_results[:top_k]]
