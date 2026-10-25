"""FP32 AdamW master weights and moments on CPU for constrained training GPUs."""
import torch


class CPUAdamW(torch.optim.Optimizer):
    """GradScaler sees GPU gradients; AdamW runs on matching FP32 CPU copies.

    No precision reduction is introduced. CPU versus CUDA arithmetic need not
    produce bitwise identical optimization trajectories. Checkpoint moments stay
    on CPU; model weights remain in the model's separately saved state dict.
    """
    def __init__(self, params, *, lr, betas=(.9,.95), eps=1e-8, weight_decay=.01):
        params=list(params)
        if not params or any(p.dtype!=torch.float32 for p in params):
            raise ValueError('nonempty FP32 trainable parameters required')
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
        self.masters=[torch.nn.Parameter(p.detach().cpu().clone()) for p in params]
        self.inner=torch.optim.AdamW(self.masters,lr=lr,betas=betas,eps=eps,
            weight_decay=weight_decay,foreach=False)

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('closure not supported')
        self.inner.zero_grad(set_to_none=True)
        for source,target in zip(self.param_groups,self.inner.param_groups):
            for key in ('lr','betas','eps','weight_decay'):target[key]=source[key]
            for device_param,master in zip(source['params'],target['params']):
                master.grad=None if device_param.grad is None else device_param.grad.detach().cpu().clone()
        self.inner.step()
        for device_param,master in zip(self.param_groups[0]['params'],self.masters):
            device_param.copy_(master)

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self,state_dict):
        self.inner.load_state_dict(state_dict)
        for source,target in zip(self.inner.param_groups,self.param_groups):
            for key,value in source.items():
                if key!='params':target[key]=value
        # A caller loads model weights before optimizer state, as in torch.
        with torch.no_grad():
            for device_param,master in zip(self.param_groups[0]['params'],self.masters):
                master.copy_(device_param.detach().cpu())


def require_rtx2060():
    import os
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID'
            or not torch.cuda.is_available() or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 2060'
            or torch.cuda.mem_get_info()[0]<2_400_000_000):
        raise ValueError('physicalGPU1 RTX2060 with2.4GBfree required; leave other services intact')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def assert_cpu_state(optimizer):
    if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in optimizer.masters):
        raise ValueError('master weights must remain FP32 on CPU')
    for state in optimizer.inner.state.values():
        if any(isinstance(v,torch.Tensor) and v.device.type!='cpu' for v in state.values()):
            raise ValueError('optimizer state unexpectedly moved to GPU')
