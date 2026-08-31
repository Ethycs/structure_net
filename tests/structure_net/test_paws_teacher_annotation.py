from experiments.structure_net.paws_teacher_annotation import request_fingerprint


def test_request_fingerprint_is_deterministic_and_prompt_sensitive() -> None:
    first = request_fingerprint("qwen3-8b", "one")
    assert first == request_fingerprint("qwen3-8b", "one")
    assert first != request_fingerprint("qwen3-8b", "two")
    assert first != request_fingerprint("other", "one")
