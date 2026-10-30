"""GPU-isolated worker launched by the project-owned Triton adapter.

The parent sets CUDA visibility before this interpreter starts. Stdout is a
JSON-lines protocol; llama.cpp diagnostics go to stderr.
"""
import json
import sys

from llama_cpp import Llama

model = Llama(model_path="/model-weights/Phi-4-GGUF/phi-4-Q4_K_M.gguf",
              n_gpu_layers=-1, main_gpu=0, split_mode=0, n_ctx=8192,
              type_k=8, type_v=8, flash_attn=True, verbose=False)
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    try:
        value = json.loads(line)
        response = model.create_chat_completion(messages=[{"role": "user", "content": value["prompt"]}],
            temperature=0.0, max_tokens=value["max_tokens"], response_format={"type": "json_object"})
        result = {"text": response["choices"][0]["message"]["content"],
                  "usage": response.get("usage"), "finish_reason": response["choices"][0].get("finish_reason")}
    except Exception as exc:
        result = {"error": type(exc).__name__ + ": " + str(exc)}
    print(json.dumps(result), flush=True)
