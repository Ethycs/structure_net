from experiments.structure_net.jspace_active_specificity import refine


def case(classification, dense):
    return {"classification": classification, "components": {"24": {"answer_donor": {"dense": {"target_success": dense}}}}}


def test_failed_dense_control_prevents_specificity_claim():
    result = refine(case("preliminary_country_CAV_and_J_advantage", .79))
    assert result["classification"] == "CAV_transfer_supported_country_specificity_unresolved_dense_answer_control"


def test_valid_control_preserves_base_classification():
    assert refine(case("preliminary_country_CAV_and_J_advantage", .8))["classification"] == "preliminary_country_CAV_and_J_advantage"


def test_amendment_never_rescues_failed_country_cav():
    result = refine(case("country_CAV_causality_not_established", 0.))
    assert result["classification"] == "country_CAV_causality_not_established"


def test_stopped_endpoint_remains_stopped():
    result = refine({"classification": "clean_competence_failed", "components": {}})
    assert result["classification"] == "clean_competence_failed"
    assert result["answer_only_dense_control_passed"] is None
