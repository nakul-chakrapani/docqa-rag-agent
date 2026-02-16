from openai import OpenAI
from app.config import settings
import time


class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.batch_size = 100


    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            for attempt in range(3):  # Retry with exponential backoff
                try:
                    response = self.client.embeddings.create(model=self.model, input=batch)
                    all_embeddings.extend([d.embedding for d in response.data])
                    break
                except Exception as e:
                    if attempt < 2: time.sleep(2 ** attempt)
                    else: raise
        return all_embeddings
