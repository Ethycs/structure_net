"""Inference-only view of an immutable tokenizer vocabulary."""


class CachedVocabulary:
    """Keep identical token validation without rebuilding a frozen vocabulary."""
    def __init__(self,tokenizer):
        self.wrapped=tokenizer
        self.vocabulary_size=len(tokenizer)

    def __len__(self):
        return self.vocabulary_size

    def __getattr__(self,name):
        return getattr(self.wrapped,name)
