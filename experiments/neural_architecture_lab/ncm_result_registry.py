"""Register completed NCM evidence conservatively, with exact and semantic readback."""
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def register_records(campaign, report, destination, records):
    for path, expected in q.read(campaign)["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("completed evidence changed: "+path)
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    collection = logger.hypotheses_collection
    if collection is None:
        raise ValueError("registry unavailable")
    for hypothesis, spec in records.items():
        receipt_path = destination / (hypothesis+"_registry_readback.json")
        json_path = Path("data/hypothesis_registry") / (hypothesis+".json")
        if receipt_path.exists() or json_path.exists():
            raise ValueError("fresh hypothesis registration required")
        record = {"id": hypothesis, **spec, "category": "architecture", "tested": True,
            "confirmed": False, "underpowered": True, "final_test_opened": False, "speed_tested": False,
            "human_rated": False, "campaign": str(campaign), "campaign_sha256": file_hash(campaign),
            "report": str(report), "report_sha256": file_hash(report)}
        logger.log_hypothesis(record)
        exact = collection.get(ids=[hypothesis], include=["metadatas"])
        collection.update(ids=[hypothesis], metadatas=[{**exact["metadatas"][0], **record}])
        exact = collection.get(ids=[hypothesis], include=["metadatas"])
        semantic = collection.query(query_texts=[record["description"]], n_results=10)
        if any(exact["metadatas"][0].get(k) != v for k, v in record.items()) or hypothesis not in semantic["ids"][0]:
            raise ValueError("registry readback failed")
        receipt = {"status": "registered_and_read_back", "record": record,
                   "exact": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
        for path in (receipt_path, json_path):
            write_json(path, receipt)
            if q.read(path) != receipt:
                raise ValueError("JSON registry readback failed")
