import numpy as np

from experiments.structure_net.paws_wavelet_router import extend_wavelet_coordinates, out_of_sample_coordinates


def test_landmarks_extend_to_their_exact_wavelet_rows_and_queries_are_finite():
    rng=np.random.default_rng(12);x=rng.normal(size=(8,4));probability=np.clip(rng.random((8,3)),.1,.9);landmarks=np.asarray([0,2,4,6]);basis=np.eye(4)
    coordinates=extend_wavelet_coordinates(x,probability,landmarks,basis,4)
    assert np.allclose(coordinates[landmarks],basis)
    assert coordinates.shape==(8,4)
    assert np.isfinite(coordinates).all()


def test_serialized_bundle_maps_unseen_inputs_without_labels_or_success():
    rng=np.random.default_rng(3);atlas_base=rng.normal(size=(40,10));atlas_success=(rng.random((40,3))>.5).astype(float);landmark_carrier=rng.normal(size=(6,13));landmark_probability=np.clip(rng.random((6,3)),.1,.9);basis=np.eye(6)
    bundle={"semantic_mean":np.zeros(10),"semantic_components":np.eye(10)[:5],"base_mean":np.zeros(10),"base_scale":np.ones(10),"competence_atlas_base":atlas_base,"competence_atlas_success":atlas_success,"landmark_carrier":landmark_carrier,"landmark_competence_probability":landmark_probability,"basis_128":basis}
    rows=[{"sentence1":"alpha beta","sentence2":"beta alpha"},{"sentence1":"one two","sentence2":"one three"}]
    coordinates=out_of_sample_coordinates(rng.normal(size=(2,10)),rows,bundle)
    assert coordinates.shape==(2,6)
    assert np.isfinite(coordinates).all()
