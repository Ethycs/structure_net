"""Cross-tokenizer candidates; target verification remains in llama.cpp.

Draft IDs are never passed as target IDs. Only newly drafted bytes are
re-tokenized, so the accepted target prefix cannot be rewritten.
"""
import time


def convert_candidate(target, draft, generated, draft_prefix, limit):
    data = draft.detokenize(generated, prev_tokens=draft_prefix)
    return target.tokenize(data, add_bos=False, special=False)[:limit]


class CrossTokenizerDraft:
    def __init__(self, target, draft, length):
        self.target, self.draft, self.length = target, draft, length
        self.prompt_ids = None
        self.prefix = None
        self.seconds = 0.0
        self.calls = self.proposed = self.native_tokens = 0

    def begin(self, prompt):
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        formatter = Jinja2ChatFormatter(
            template=self.draft.metadata["tokenizer.chat_template"],
            eos_token=self.draft.detokenize([self.draft.token_eos()], special=True).decode(),
            bos_token=self.draft.detokenize([self.draft.token_bos()], special=True).decode())
        self.prefix = formatter(messages=[{"role": "user", "content": prompt}]).prompt.encode()
        self.prompt_ids = None
        self.seconds = 0.0
        self.calls = self.proposed = self.native_tokens = 0
        self.draft.reset()

    def __call__(self, input_ids):
        import numpy as np
        start = time.perf_counter()
        if self.prompt_ids is None:
            raise RuntimeError("target prefill was not captured")
        if input_ids[:len(self.prompt_ids)].tolist() != self.prompt_ids:
            raise RuntimeError("confirmed target prompt changed")
        continuation = self.target.detokenize(input_ids[len(self.prompt_ids):].tolist(), prev_tokens=self.prompt_ids)
        draft_ids = self.draft.tokenize(self.prefix + continuation, add_bos=True, special=True)
        generator = self.draft.generate(draft_ids, temp=0.0, repeat_penalty=1.0)
        generated = []
        try:
            for _ in range(self.length):
                token = next(generator)
                if token == self.draft.token_eos():
                    break
                generated.append(token)
        finally:
            generator.close()
        candidates = convert_candidate(self.target, self.draft, generated, draft_ids, self.length)
        self.calls += 1
        self.proposed += len(candidates)
        self.native_tokens += len(generated)
        self.seconds += time.perf_counter() - start
        return np.asarray(candidates, dtype=np.intc)
