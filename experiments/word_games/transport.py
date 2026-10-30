"""Exact-model, no-fallback Triton transport for the cached Phi-4 14B."""
import json
import time
import requests

from .common import write

MODEL = "word-games-phi4-14b"
BASE_URL = "http://127.0.0.1:8000"


class TransportFailure(RuntimeError):
    pass


class Phi4:
    def __init__(self, base_url=BASE_URL, model=MODEL):
        if model != MODEL:
            raise ValueError("WORD_GAMES requires the specified Phi-4 14B model")
        self.base_url, self.model = base_url.rstrip("/"), model

    def complete(self, prompt, path, *, max_tokens=1024):
        body = {"inputs": [
            {"name": "text_input", "shape": [1], "datatype": "BYTES", "data": [prompt]},
            {"name": "sampling_parameters", "shape": [1], "datatype": "BYTES", "data": [json.dumps({
                "temperature": 0.0, "max_tokens": max_tokens, "response_format": {"type": "json_object"}})]}]}
        write(path / "request.json", {"model": self.model, "body": body})
        start = time.monotonic()
        try:
            response = requests.post(f"{self.base_url}/v2/models/{self.model}/infer", json=body, timeout=240)
            record = {"http_status": response.status_code, "body": response.text,
                      "elapsed_seconds": time.monotonic() - start, "model": self.model}
            write(path / "response.json", record)
            response.raise_for_status()
            wire = response.json()
            if wire.get("model_name") != self.model:
                raise ValueError("response came from another model")
            return next(x["data"][0] for x in wire["outputs"] if x["name"] == "text_output")
        except Exception as exc:
            if not (path / "response.json").exists():
                write(path / "response.json", {"error_type": type(exc).__name__, "error": str(exc),
                      "elapsed_seconds": time.monotonic() - start, "model": self.model})
            raise TransportFailure(f"Phi-4 transport failed: {type(exc).__name__}") from exc
