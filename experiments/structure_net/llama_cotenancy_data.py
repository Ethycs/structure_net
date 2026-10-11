"""Deterministic exact-source summary assay; deliberately not application data."""
import random
import re
from transformers import AutoTokenizer
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, write_json

SEEDS = (7, 17, 29, 41, 53)


def example(seed, family, split):
    rng = random.Random(f"llama-cotenancy-{split}-{seed}-{family}")
    people = rng.sample(["Amira", "Bruno", "Chen", "Daria", "Elena", "Farid", "Grace", "Hugo", "Inez", "Jonah"], 2)
    project = rng.choice(["Orion review", "Cedar launch", "Atlas briefing", "Willow workshop"]) if family == "work" else rng.choice(["book club", "garden visit", "birthday lunch", "weekend picnic"])
    dates = rng.sample(["May 3", "May 11", "June 8", "June 19", "July 6", "July 21", "August 4", "August 16"], 2)
    locations = rng.sample(["the library", "Riverside Hall", "the community center", "Maple Cafe", "the north garden", "Room 204"], 2)
    gold = [f"The final date for the {project} is {dates[1]}.",
            f"{people[0]} is the confirmed coordinator for the {project}.",
            f"The confirmed venue for the {project} is {locations[1]}."]
    sentences = [f"{people[1]} initially proposed {dates[0]} for the {project}.", gold[0],
                 f"{people[1]} was first suggested as coordinator.", gold[1],
                 f"The group discussed meeting at {locations[0]}.", gold[2],
                 "Budget discussion is postponed until next month.", "This note records the final decisions after all revisions."]
    rng.shuffle(sentences)
    source = "\n".join(sentences)
    prompt = "Copy exactly the three source sentences stating the final date, confirmed coordinator, and confirmed venue. Output one sentence per line. Omit superseded proposals and unrelated details. Do not paraphrase.\n\nSOURCE:\n" + source
    return {"id": f"{split}_{family}_{seed}", "split": split, "seed": seed, "family": family,
            "source": source, "source_sentences": sentences, "gold_fragments": gold, "prompt": prompt}


def score(text, row):
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip() for line in text.splitlines() if line.strip()]
    covered = [fragment in text for fragment in row["gold_fragments"]]
    supported = [line in row["source_sentences"] for line in lines]
    return {"fragment_coverage": sum(covered)/len(covered), "covered_fragments": sum(covered),
            "total_fragments": len(covered), "supported_line_rate": sum(supported)/max(1, len(lines)),
            "supported_lines": sum(supported), "output_lines": len(lines), "complete_summary": all(covered) and bool(lines) and all(supported),
            "fragment_matches": covered, "line_matches": supported}


def prepare(path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    rows = [example(seed, family, "validation") for seed in (1963, 1964) for family in ("work", "personal")]
    rows += [example(seed, family, "test") for seed in SEEDS for family in ("work", "personal")]
    for row in rows:
        messages = [{"role": "system", "content": "You extract faithful summaries. Output only source sentences, one per line."},
                    {"role": "user", "content": row["prompt"]}]
        row["prompt_ids"] = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        row["reference_ids"] = tokenizer.encode("\n".join(row["gold_fragments"]), add_special_tokens=False) + [128009]
    value = {"schema": "nal.llama-cotenancy-extractive-fixtures.v1", "generator": "llama_cotenancy_data.py", "examples": rows,
             "benchmark_ids": [(row["prompt_ids"]*4)[:512] for row in rows[:4]], "eos_ids": [128001, 128008, 128009]}
    write_json(path, value)
    return value
