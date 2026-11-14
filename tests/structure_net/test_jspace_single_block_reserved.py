from types import SimpleNamespace

import pytest

from experiments.structure_net import jspace_single_block_reserved as reserved


class FakeCuda:
    def __init__(self,backend="native",initialized=False):
        self.events=[];self.initialized=initialized;self.memory=SimpleNamespace(get_allocator_backend=lambda:backend)
    def is_initialized(self):return self.initialized
    def get_device_properties(self,index):
        self.events.append(("properties",index));self.initialized=True
        return SimpleNamespace(total_memory=12*1024**3)
    def set_per_process_memory_fraction(self,fraction,device):self.events.append(("fraction",fraction,device))


def test_cap_is_lease_minus256mib_divided_by_physical_memory():
    fake=FakeCuda();lease=dict(id="test",allocation={"GPU-test":7168})
    policy=dict(headroom_mib=256,allocator_backend="native")
    value=reserved.allocator_settings(lease,policy,cuda=fake,visible="GPU-test")
    assert value["allocator_limit_bytes"]==(7168-256)*1024**2
    assert value["memory_fraction"]==pytest.approx(6912/12288)
    assert fake.events==[("properties",0),("fraction",value["memory_fraction"],0)]


def test_invalid_or_preinitialized_configuration_rejected():
    policy=dict(headroom_mib=256,allocator_backend="native")
    lease=dict(id="test",allocation={"GPU-test":7168})
    with pytest.raises(ValueError,match="Single visible"):
        reserved.allocator_settings(lease,policy,cuda=FakeCuda(),visible="")
    with pytest.raises(RuntimeError,match="fresh CUDA"):
        reserved.allocator_settings(lease,policy,cuda=FakeCuda(initialized=True),visible="GPU-test")
    with pytest.raises(RuntimeError,match="native PyTorch"):
        reserved.allocator_settings(lease,policy,cuda=FakeCuda(backend="cudaMallocAsync"),visible="GPU-test")
    with pytest.raises(ValueError,match="headroom"):
        reserved.allocator_settings(dict(id="x",allocation={"GPU-test":128}),policy,cuda=FakeCuda(),visible="GPU-test")


def mock_prepared(tmp_path,monkeypatch):
    root=tmp_path/"study"
    def prepare(path,*_):
        path.mkdir();reserved.single.recipe.write(path/"plan.json",dict(fingerprint="base",schema=reserved.single.SCHEMA))
        return dict(status="prepared",fingerprint="base")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(reserved.single,"prepare",prepare)
    monkeypatch.setattr(reserved.single,"verify",lambda p:reserved.single.recipe.read(p/"plan.json"))
    reserved.prepare(root,tmp_path,tmp_path,tmp_path)
    return root


def test_policy_is_bound_to_original_plan_and_runtime_source(tmp_path,monkeypatch):
    root=mock_prepared(tmp_path,monkeypatch)
    plan,policy=reserved.verify(root)
    assert plan["schema"]==reserved.single.SCHEMA and policy["headroom_mib"]==256
    assert (root/"runtime_producer/experiments/structure_net/jspace_single_block_reserved.py").exists()
    changed=reserved.single.recipe.read(root/"runtime_policy.json");changed["headroom_mib"]=0
    (root/"runtime_policy.json").write_text(__import__('json').dumps(changed))
    monkeypatch.setattr(reserved,"reservation",lambda:pytest.fail("Lease acquired before policy validation"))
    with pytest.raises(ValueError,match="allocator policy"):reserved.run(root,"pilot")


def test_lease_then_allocator_then_unchanged_runner_order_and_receipt(tmp_path,monkeypatch):
    root=mock_prepared(tmp_path,monkeypatch);events=[]
    lease=dict(id="test",allocation={"GPU-test":7168})
    monkeypatch.setattr(reserved,"reservation",lambda:events.append("lease") or lease)
    def settings(*_):events.append("allocator");return dict(allocator_limit_bytes=6912*1024**2)
    monkeypatch.setattr(reserved,"allocator_settings",settings)
    def run(path,stage):
        events.append("trainer");(path/stage).mkdir()
        reserved.single.recipe.write(path/stage/"manifest.json",dict(status="completed",fingerprint="base",peak_reserved_bytes=100))
        return dict(status="completed",systems_only=True)
    monkeypatch.setattr(reserved.single,"run",run)
    result=reserved.run(root,"pilot")
    assert events==["lease","allocator","trainer"] and result["runtime"]["status"]=="completed"
    plan,policy=reserved.verify(root);reserved.verify_runtime_receipt(root,"pilot",plan,policy)


def test_denied_lease_never_calls_allocator_and_records_failure(tmp_path,monkeypatch):
    root=mock_prepared(tmp_path,monkeypatch)
    def denied():raise RuntimeError("lease denied")
    monkeypatch.setattr(reserved,"reservation",denied)
    monkeypatch.setattr(reserved,"allocator_settings",lambda *_:pytest.fail("CUDA accessed without lease"))
    with pytest.raises(RuntimeError,match="lease denied"):reserved.run(root,"pilot")
    result=reserved.single.recipe.read(root/"runtime_pilot.json")
    assert result["status"]=="failed" and result["lease_id"] is None
