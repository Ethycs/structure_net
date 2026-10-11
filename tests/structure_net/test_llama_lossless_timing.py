from experiments.structure_net import run_llama_lossless_timing as study


def test_two_arm_order_actually_alternates(monkeypatch):
    calls=[]
    def arm(name):
        def operation():
            calls.append(name)
        return operation
    def timed(fn):
        value=fn()
        return {"wall_ms":1.0,"cuda_ms":1.0},value
    monkeypatch.setattr(study.common,"timed",timed)
    results,orders=study.paired({"original":arm("original"),"residual":arm("residual")})
    assert calls[:2]==["original","residual"]
    assert orders==[["original","residual"],["residual","original"],["original","residual"],["residual","original"],["original","residual"]]
    assert calls[2:]==[name for order in orders for name in order]
    assert all(len(value["samples"])==5 for value in results.values())
