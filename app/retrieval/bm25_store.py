from rank_bm25 import BM25Okapi
import pickle, os, re


class BM25Store:
    def __init__(self, index_path="data/bm25_index.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []


    def _tokenize(self, text):
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in text.split() if len(t) > 2]


    def add_documents(self, chunks: list[dict]):
        for chunk in chunks:
            self.documents.append(chunk)
            self.tokenized_docs.append(self._tokenize(chunk["text"]))
        self.bm25 = BM25Okapi(self.tokenized_docs)


    def search(self, query, top_k=20, doc_filter=None):
        scores = self.bm25.get_scores(self._tokenize(query))
        scored = [(i, s) for i, s in enumerate(scores) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, score) in enumerate(scored[:top_k], 1):
            doc = self.documents[idx]
            if doc_filter and doc.get("doc_id") != doc_filter: continue
            results.append({**doc, "bm25_score": score, "bm25_rank": rank})
        return results


    def save(self): pickle.dump({"docs": self.documents, "tok": self.tokenized_docs}, open(self.index_path, "wb"))
    def load(self):
        if os.path.exists(self.index_path):
            data = pickle.load(open(self.index_path, "rb"))
            self.documents, self.tokenized_docs = data["docs"], data["tok"]
            self.bm25 = BM25Okapi(self.tokenized_docs)
            