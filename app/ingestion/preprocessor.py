import tiktoken, re

class TextPreprocessor:
    def __init__(self, model="text-embedding-3-small"):
        self.encoder = tiktoken.encoding_for_model(model)


    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))


    def clean_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()
