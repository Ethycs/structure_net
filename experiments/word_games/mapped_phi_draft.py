"""Opt-in token-table draft adapter; the target must verify every proposal."""
import time

from .phi_token_map import TokenMapBundle
from .speculative_draft import CrossTokenizerDraft


class MappedPhiDraft(CrossTokenizerDraft):
    def __init__(self, target, draft, length, map_directory, *, mapped_conditioning=False):
        super().__init__(target, draft, length)
        from .build_phi_token_map import snapshot_model
        self.maps = TokenMapBundle.load(map_directory)
        self.maps.validate_tokenizers(snapshot_model(draft, "small"), snapshot_model(target, "large"))
        self.mapped_conditioning = mapped_conditioning

    def begin(self, prompt):
        super().begin(prompt)
        self.prefix_ids = self.draft.tokenize(self.prefix, add_bos=True, special=True)

    def __call__(self, input_ids):
        import numpy as np
        start = time.perf_counter()
        if self.prompt_ids is None or input_ids[:len(self.prompt_ids)].tolist() != self.prompt_ids:
            raise RuntimeError("confirmed target prompt changed or was not captured")
        continuation = input_ids[len(self.prompt_ids):].tolist()
        if self.mapped_conditioning:
            # Exact bytes, potentially noncanonical draft segmentation. This
            # changes the draft's conditioning; target verification is mandatory.
            draft_ids = self.prefix_ids + self.maps.large_to_small.translate(continuation)
        else:
            data = self.maps.large_to_small.source.decode(continuation)
            draft_ids = self.draft.tokenize(self.prefix + data, add_bos=True, special=True)
        generator = self.draft.generate(draft_ids, temp=0.0, repeat_penalty=1.0)
        generated = []
        try:
            for _ in range(self.length):
                token = next(generator)
                if not self.maps.small_to_large.source.ordinary(token):
                    break  # EOS/role/control tokens are never translated as text.
                generated.append(token)
        finally:
            generator.close()
        candidates = self.maps.small_to_large.translate(generated, limit=self.length)
        self.calls += 1
        self.proposed += len(candidates)
        self.native_tokens += len(generated)
        self.seconds += time.perf_counter() - start
        return np.asarray(candidates, dtype=np.intc)
