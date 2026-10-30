"""CPU-only readback of native checkpoint results; no inference or judging."""
import json
import os
from pathlib import Path

from experiments.structure_net.ncm_pretrained_capability import BERT, ROOT, sha, read_pickle, write


def validate_text(row):
    """Independently validate text formatting, constraints and original clues."""
    text = row["text"]
    lines = text.split("|")
    if len(lines) != 9 or any(len(line) != 9 or not set(line) <= set("123456789") for line in lines):
        return False
    units = lines + ["".join(line[c] for line in lines) for c in range(9)]
    units += ["".join(lines[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3))
              for br in (0, 3, 6) for bc in (0, 3, 6)]
    clues = [v for v in row["puzzle"][1:-1] if v != 10]
    return (all(sorted(unit) == list("123456789") for unit in units)
            and all(not clue or int(cell) == clue for cell, clue in zip("".join(lines), clues, strict=True)))


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    from transformers import AutoTokenizer
    out = ROOT / "readback"
    out.mkdir(parents=True, exist_ok=False)
    (out / "producer_source.py").write_bytes(Path(__file__).read_bytes())
    results = {}
    for task in ("easy", "hard", "lm1b"):
        stage = ROOT / f"{task}_full"
        rows = json.loads((stage / "outputs.json").read_text())
        result = json.loads((stage / "result.json").read_text())
        author = read_pickle(stage / "author_output.pickle")
        assert [r["text"] for r in rows] == author["hyps"]
        assert len(rows) == result["n"]
        for filename, digest in json.loads((stage / "producer.json").read_text())["hashes"].items():
            assert sha(filename) == digest
        if task != "lm1b":
            valid = sum(validate_text(row) for row in rows)
            exact = sum(h == r for h, r in zip(author["hyps"], author["refs"], strict=True))
            assert valid == exact == result["counts"]["solved"] == result["counts"]["exact"]
            assert all(validate_text(r) == r["scores"]["solved"] for r in rows)
            results[task] = {"n": len(rows), "exact": exact, "validated": valid,
                             "solved_percent": 100 * exact / len(rows), "all_rows_replayed": True}
        else:
            assert author["raw_ids"] == [r["raw_ids"] for r in rows]
            tokenizer = AutoTokenizer.from_pretrained(BERT, local_files_only=True)
            display = []
            for row in rows:
                ids = row["raw_ids"]
                # Preserve the full canvas; CLS separates multiple sentences here.
                display.append({"index": row["index"], "author_hypothesis": row["text"],
                    "raw_text_including_specials": tokenizer.decode(ids, skip_special_tokens=False),
                    "full_canvas_without_specials": tokenizer.decode(ids, skip_special_tokens=True),
                    "cls_positions": [i for i, v in enumerate(ids) if v == 101]})
            write(out / "lm1b_texts.json", display)
            with (out / "lm1b_samples.txt").open("x") as f:
                f.write("All 16 fixed seed-42 samples in generation order. No selection.\n"
                        "Full 128-position canvases, including CLS sentence boundaries.\n"
                        "The author's separate first-EOS display field is empty in every case.\n\n")
                for row in display:
                    f.write(f"Sample {row['index']}\n{row['raw_text_including_specials']}\n\n")
            results[task] = {"n": len(rows), "starts_with_CLS": sum(r["raw_ids"][0] == 101 for r in rows),
                            "empty_author_hypotheses": sum(not r["text"] for r in rows),
                            "nonempty_full_canvases": sum(bool(r["full_canvas_without_specials"].strip()) for r in display),
                            "accuracy": None, "generative_perplexity": None}
            for row in display:
                print(json.dumps(row, ensure_ascii=False), flush=True)
        results[task]["raw_outputs_sha256"] = sha(stage / "outputs.json")
    write(out / "summary.json", results)
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
