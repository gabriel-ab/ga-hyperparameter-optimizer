import math


class TfidfVectorizer:
    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self.word_inverse_document_frequency = {}

    def fit(self, documents: list[str]) -> None:
        n_documents = len(documents)
        self._init_word_inverse_document_frequency(documents)
        for term, count in self.word_inverse_document_frequency.items():
            self.word_inverse_document_frequency[term] = math.log(n_documents / count)
    
    def _init_word_inverse_document_frequency(self, documents: list[str]):
        for document in documents:
            self.word_inverse_document_frequency["<OOV>"] = 1
            seen = set()
            for term in document.split():
                if term not in seen:
                    self.word_inverse_document_frequency.setdefault(term, 0)
                    self.word_inverse_document_frequency[term] += 1
                    seen.add(term)
        

    def encode(self, documents: list[str]) -> list[list[float]]:
        encodings = []
        for document in documents:

            terms: list = document.split()

            if len(terms) < self.max_len:
               print("test", len(terms))
               pad = ["<OOV>"] * int(self.max_len - len(terms))
               terms.extend(pad)

            encoded_document = []
            for term in terms:
                encoded_term = self.word_inverse_document_frequency[term]
                encoded_document.append(encoded_term)
            encodings.append(encoded_document[:self.max_len])
        return encodings


if __name__ == "__main__":
    import pandas as pd

    from aes.cleaners import TextCleaner


    dataset = pd.read_parquet("data/dataset.parquet")
    vectorizer = TfidfVectorizer(max_len=40)
    cleaner = TextCleaner()
    texts = [cleaner(row["text"]) for _, row in dataset.iterrows()]
    vectorizer.fit(texts)
    encoded = vectorizer.encode([texts[0]])
    encoded2 = vectorizer.encode([texts[1]])
    encodeds = vectorizer.encode(texts)
    print(len(encoded[0]))
    print(len(encoded2[0]))
    print([len(x) for x in encodeds])
