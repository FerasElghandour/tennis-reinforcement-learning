import numpy as np
import pytest
from pathlib import Path

torch=pytest.importorskip('torch')
from policy import load_policy, predict, export_policy


def test_saved_actor_supports_variable_batch_sizes():
    checkpoint=Path(__file__).resolve().parents[1]/'run-seed-0/checkpoint_actor.pth'
    actor=load_policy(checkpoint)
    for size in [1,2,5]:
        actions=np.asarray(predict(actor,np.ones((size,24))))
        assert actions.shape==(size,2) and np.isfinite(actions).all() and np.abs(actions).max()<=1


@pytest.mark.parametrize('values',[[], [[0]*23], [[float('nan')]*24], [0]*24])
def test_bad_observation_shapes_and_nonfinite_values_are_rejected(values):
    with pytest.raises(ValueError,match='Observations'):predict(None,values)


def test_exported_policy_matches_saved_actor_on_new_inputs(tmp_path):
    checkpoint=Path(__file__).resolve().parents[1]/'run-seed-0/checkpoint_actor.pth'
    output=tmp_path/'actor.pt2';metadata=export_policy(checkpoint,output)
    actor=load_policy(checkpoint);exported=torch.export.load(str(output)).module()
    states=np.random.default_rng(42).normal(size=(5,24)).astype(np.float32)
    with torch.inference_mode():actual=exported(torch.from_numpy(states)).numpy()
    np.testing.assert_allclose(actual,np.asarray(predict(actor,states)),rtol=1e-5,atol=1e-6)
    assert len(metadata['source_sha256'])==64 and output.with_suffix('.json').exists()
