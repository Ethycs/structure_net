"""Exact-arithmetic CUDA graph replay; no weight compression or reassociation."""
import time
import torch


def clone_tree(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    if value is None:
        return None
    raise TypeError(type(value))


def bitwise_tree(actual, expected):
    if torch.is_tensor(expected):
        return (torch.is_tensor(actual) and actual.shape == expected.shape and actual.dtype == expected.dtype
                and torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))
    if expected is None:
        return actual is None
    return type(actual) is type(expected) and len(actual) == len(expected) and all(bitwise_tree(a, b) for a, b in zip(actual, expected))


class CapturedCall:
    def __init__(self, function, example):
        if example.device.type != "cuda":
            raise ValueError("CUDA input required")
        self.input = example.clone()
        self.function = function
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        started = time.perf_counter()
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = function(self.input)
                del output
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = function(self.input)
        torch.cuda.synchronize()
        self.capture_and_warmup_seconds = time.perf_counter() - started

    def __call__(self, value):
        if value.shape != self.input.shape or value.dtype != self.input.dtype or value.device != self.input.device:
            raise ValueError("graph input shape/dtype/device changed")
        self.input.copy_(value)
        self.graph.replay()
        return clone_tree(self.output)
