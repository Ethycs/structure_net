"""Move task instructions after unchanged source; preserve failed prompt evidence."""
import argparse
import json
from pathlib import Path

from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, sha256, write_json


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--drafts", type=int, nargs="+", default=[0, 8])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve immutable fixtures")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    value = json.loads(args.input.read_text())
    value.update(parent_fixture=str(args.input), parent_sha256=sha256(args.input),
                 prompt_revision=2, repeats=args.repeats, drafts=args.drafts,
                 prompt_producer_sha256=sha256(__file__))
    for example in value["examples"]:
        body = "\n\n".join(f"[{f['id']}]\n{f['source']}" for f in example["fragments"])
        instruction = ("\n\nTASK: Summarize only the final approved decisions in each fragment above. "
                       "Include the project, final owner, approved budget and deadline. "
                       "Ignore superseded proposals and background discussion. "
                       "Write one short paragraph per fragment, labeled with its ID, at most 128 tokens each. "
                       "Do not create additional fragments. Output only the summaries.")
        messages = [{"role": "system", "content": "You write accurate concise summaries grounded in the supplied evidence."},
                    {"role": "user", "content": body + instruction}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if len(ids) > 6300:
            raise ValueError("revised prompt exceeds raw limit; no truncation is allowed")
        example.update(prompt_ids=ids, prompt_tokens=len(ids))
    write_json(args.output, value)
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "lengths": [e["prompt_tokens"] for e in value["examples"]]}))


if __name__ == "__main__":
    main()
