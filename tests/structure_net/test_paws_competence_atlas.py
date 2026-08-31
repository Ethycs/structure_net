import json
from experiments.structure_net.paws_competence_atlas import best_checkpoint


def test_best_checkpoint_uses_balanced_accuracy_then_accuracy_then_id(tmp_path):
    path=tmp_path/"campaign.json"
    path.write_text(json.dumps({"results":{"z":{"metrics":{"balanced_accuracy":.8,"accuracy":.7},"checkpoint":"z.pt","error":None},"a":{"metrics":{"balanced_accuracy":.8,"accuracy":.7},"checkpoint":"a.pt","error":None},"b":{"metrics":{"balanced_accuracy":.7,"accuracy":.9},"checkpoint":"b.pt","error":None}}}))
    assert best_checkpoint(path).name=="a.pt"
