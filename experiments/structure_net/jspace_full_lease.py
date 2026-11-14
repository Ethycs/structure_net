"""Pause, without GPU allocation, while an active reservation settles."""
import json
import time

from structure_net_resources.client import require_gpu_lease,ReservationError
from .jspace_full_capture import Stage as CapturedStage


def wait_ready(device_ids=None):
    deadline=time.monotonic()+45
    while True:
        try:return require_gpu_lease(device_ids)
        except ReservationError as error:
            if str(error)!="GPU lease is active but its memory is not ready" or time.monotonic()>=deadline:raise
            print(json.dumps({"kind":"reservation_wait","reason":str(error),"gpu_work_paused":True}),flush=True)
            time.sleep(3)


class StableStage(CapturedStage):
    def __init__(self,*args,**kwargs):
        wait_ready([0])
        super().__init__(*args,**kwargs)
        self.guard=wait_ready
