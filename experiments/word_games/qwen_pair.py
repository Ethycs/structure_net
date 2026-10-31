"""Pinned Qwen2.5 14B target / 3B draft for the active WORD_GAMES work."""
from pathlib import Path

MODEL = "word-games-qwen2.5-14b"
HYPOTHESIS = "word-games-qwen25-pair-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-31_word-games-qwen-pair-plan.md")
MODEL_ROOT = Path("/home/rabbit/models")
PAIRS = {
    "target": {
        "repository": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "revision": "b466e1f8c07172155743e8e1307507d8a4f91fbd",
        "files": [
            ("qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf", 3991999872, "a09ea5e7b1eafb1b30b241726c3cc3c905c96f14ad41e246ffa5f44e53904f68"),
            ("qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf", 3989373504, "21b9457d079680d284e90ef69607c4b2d8ef64a09d4729cb7b5e1357bdba41ae"),
            ("qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf", 1006737120, "c8d37006760a387a35216e070e6664d7da927f10be8eb870fef2e3d4833d9976"),
        ],
    },
    "draft": {
        "repository": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "revision": "7dabda4d13d513e3e842b20f0d435c732f172cbe",
        "files": [("qwen2.5-3b-instruct-q4_k_m.gguf", 2104932768, "626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d")],
    },
}


def directory(role):
    return MODEL_ROOT / PAIRS[role]["repository"].split("/")[1]


def entrypoint(role):
    return directory(role) / PAIRS[role]["files"][0][0]
