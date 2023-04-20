import re


class TextCleaner:
    def __call__(self, text: str) -> str:
        uncesored_text = self._remove_censored(text)
        unpuctuated_text = self._remove_punctuation(uncesored_text)
        return unpuctuated_text.lower()

    def _remove_censored(self, text: str) -> str:
        return re.sub(r"@[A-Z]+[0-9]*", "", text)
    
    def _remove_punctuation(self, text: str) -> str:
        return re.sub(r"[^\w\s]", "", text)
