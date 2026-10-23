"""Residual concepts over native embeddings; no learned text decoder."""
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ConceptConfig:
    canvas: int = 128
    levels: tuple = (4,8,16,32,64)
    entries: int = 256
    iterations: int = 32
    chunk: int = 1024
    initialization_seed: int = 3042

    def __post_init__(self):
        if (self.canvas<1 or not self.levels or self.entries<2 or self.iterations<1 or self.chunk<1
            or any(k<1 or self.canvas%k for k in self.levels) or list(self.levels)!=sorted(set(self.levels))):
            raise ValueError("invalid native concept hierarchy")


def pool(x, width):
    if x.ndim != 3 or x.shape[1]%width:
        raise ValueError("equal native-token cells required")
    return x.reshape(x.shape[0],width,x.shape[1]//width,x.shape[2]).mean(2)


def expand(x, canvas):
    if x.ndim != 3 or canvas%x.shape[1]:
        raise ValueError("invalid concept expansion")
    return x.repeat_interleave(canvas//x.shape[1],dim=1)


def nearest(x, book, chunk=1024):
    if x.ndim != 2 or book.ndim != 2 or x.shape[1] != book.shape[1]:
        raise ValueError("concept distance shape mismatch")
    result = []
    # Explicitly disable ambient autocast: code language is always FP32.
    with torch.autocast(x.device.type,enabled=False):
        book = book.float(); norms = book.square().sum(-1)
        for i in range(0,len(x),chunk):
            values = x[i:i+chunk].float()
            distance = values.square().sum(-1,keepdim=True)+norms[None]-2*values@book.T
            if not torch.isfinite(distance).all(): raise ValueError("nonfinite concept distances")
            result.append(distance.argmin(-1))
    return torch.cat(result)


def fit_book(vectors_cpu, config, level, device):
    if len(vectors_cpu)<config.entries or vectors_cpu.ndim != 2:
        raise ValueError("enough distinct training-vector indices required")
    generator = torch.Generator().manual_seed(config.initialization_seed+level)
    pick = torch.randperm(len(vectors_cpu),generator=generator)[:config.entries-1]
    book = torch.cat((torch.zeros_like(vectors_cpu[:1]),vectors_cpu[pick]),0).to(device).float()
    for _ in range(config.iterations):
        sums = torch.zeros_like(book); counts = torch.zeros(len(book),device=device,dtype=torch.long)
        for i in range(0,len(vectors_cpu),config.chunk):
            x = vectors_cpu[i:i+config.chunk].to(device).float()
            ids = nearest(x,book,config.chunk)
            sums.index_add_(0,ids,x); counts.index_add_(0,ids,torch.ones_like(ids))
        active = counts>0; active[0] = False
        book[active] = sums[active]/counts[active,None]
        book[0].zero_()
        if not torch.isfinite(book).all(): raise ValueError("nonfinite concept centers")
    return book


def encode(target, books, config):
    if target.ndim != 3 or target.shape[1] != config.canvas or len(books) != len(config.levels):
        raise ValueError("native concept target/book shape mismatch")
    reconstruction = torch.zeros_like(target,dtype=torch.float32)
    codes,errors = [],[]
    for width,book in zip(config.levels,books):
        residual = pool(target.float()-reconstruction,width)
        ids = nearest(residual.flatten(0,1),book,config.chunk).reshape(len(target),width)
        reconstruction = reconstruction+expand(book[ids],config.canvas)
        codes.append(ids); errors.append((target.float()-reconstruction).square().mean((1,2)))
    return codes,reconstruction,torch.stack(errors,1)


def conditional_target(target, reconstruction, book, width, chunk=1024):
    residual = pool(target.float()-reconstruction.float(),width)
    return nearest(residual.flatten(0,1),book,chunk).reshape(len(target),width)


def check_repair(target, books, config):
    clean,_,errors = encode(target,books,config)
    initial = target.float().square().mean((1,2))
    if ((errors-torch.cat((initial[:,None],errors[:,:-1]),1))>1e-6).any():
        raise ValueError("zero-code monotone error invariant failed")
    cumulative = torch.zeros_like(target,dtype=torch.float32)
    for width,book,labels in zip(config.levels,books,clean):
        if not torch.equal(conditional_target(target,cumulative,book,width,config.chunk),labels):
            raise ValueError("clean-history labels differ")
        cumulative = cumulative+expand(book[labels],config.canvas)
    # Target-independent wrong roots; compare experts at the SAME history.
    wrong = (clean[0]+1)%config.entries
    cumulative = expand(books[0][wrong],config.canvas)
    gains = []
    for width,book,static in zip(config.levels[1:],books[1:],clean[1:]):
        repaired = conditional_target(target,cumulative,book,width,config.chunk)
        a = cumulative+expand(book[static],config.canvas)
        b = cumulative+expand(book[repaired],config.canvas)
        gain = (target.float()-a).square().mean((1,2))-(target.float()-b).square().mean((1,2))
        if (gain < -1e-6).any(): raise ValueError("conditional expert worse at same history")
        gains.append(gain); cumulative = b
    return {"zero_code_monotone":True,"clean_history_label_parity":True,
        "conditional_expert_not_worse_at_same_history":True,"mean_repair_gain":float(torch.stack(gains).mean())}
