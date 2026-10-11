"""Portable, bit-preserving CPU archives; decoding restores dense tensors.

No floating-point arithmetic is used. Entropy coding changes storage only.
The manifest retains typed checkpoint metadata and tensor alias references.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import zstandard as zstd

SCHEMA = "structure-net.lossless-tensor-archive.v1"
LAYOUTS = ("raw", "byte_shuffle", "bit_shuffle", "xor_bit_shuffle")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def shuffle(raw, width, layout):
    if layout not in LAYOUTS or width < 1 or len(raw) % width:
        raise ValueError("invalid layout or element alignment")
    if layout == "raw" or not raw:
        return raw
    values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, width)
    if layout == "byte_shuffle":
        return values.T.tobytes()
    if layout == "xor_bit_shuffle":
        values = values.copy()
        values[1:] = np.bitwise_xor(values[1:], values[:-1])
    planes = np.unpackbits(values, axis=1, bitorder="little")
    return np.packbits(planes, axis=0, bitorder="little").T.tobytes()


def unshuffle(encoded, width, count, layout):
    if layout not in LAYOUTS or width < 1 or count < 0:
        raise ValueError("invalid layout or element count")
    if not count:
        if encoded:
            raise ValueError("nonempty encoding of empty tensor")
        return b""
    expected = width * (count if layout in ("raw", "byte_shuffle") else 8 * ((count + 7) // 8))
    if len(encoded) != expected:
        raise ValueError("invalid transformed byte count")
    if layout == "raw":
        return encoded
    if layout == "byte_shuffle":
        return np.frombuffer(encoded, np.uint8).reshape(width, count).T.tobytes()
    packed = np.frombuffer(encoded, np.uint8).reshape(width * 8, -1).T
    planes = np.unpackbits(packed, axis=0, count=count, bitorder="little")
    values = np.packbits(planes, axis=1, bitorder="little")
    if layout == "xor_bit_shuffle":
        values = np.bitwise_xor.accumulate(values, axis=0)
    return values.tobytes()


def encode_chunk(raw, width, layout, level=3):
    encoded = zstd.ZstdCompressor(level=level, threads=0).compress(shuffle(raw, width, layout))
    return (encoded, True) if len(encoded) < len(raw) else (raw, False)


def decode_chunk(encoded, width, count, layout, compressed):
    if not compressed:
        if len(encoded) != width * count:
            raise ValueError("invalid raw chunk size")
        return encoded
    limit = width * (count if layout in ("raw", "byte_shuffle") else 8 * ((count + 7) // 8))
    transformed = zstd.ZstdDecompressor().decompress(encoded, max_output_size=limit)
    return unshuffle(transformed, width, count, layout)


def flatten(value):
    tensors, references = [], {}

    def visit(v, path):
        if torch.is_tensor(v):
            if id(v) not in references:
                references[id(v)] = len(tensors)
                tensors.append((path, v))
            return ["tensor", references[id(v)]]
        if isinstance(v, dict):
            return ["dict", [[visit(k, path + "/key"), visit(x, path + "/" + str(k))] for k, x in v.items()]]
        if isinstance(v, (list, tuple)):
            return ["tuple" if isinstance(v, tuple) else "list", [visit(x, path + "/" + str(i)) for i, x in enumerate(v)]]
        if v is None or isinstance(v, (bool, str, int, float)):
            return ["scalar", v]
        raise TypeError(f"unsupported metadata at {path}: {type(v)}")

    return visit(value, "root"), tensors


def inflate(tree, tensors):
    tag, value = tree
    if tag == "tensor":
        return tensors[value]
    if tag == "scalar":
        return value
    if tag == "dict":
        return {inflate(k, tensors): inflate(v, tensors) for k, v in value}
    if tag in ("list", "tuple"):
        values = [inflate(x, tensors) for x in value]
        return tuple(values) if tag == "tuple" else values
    raise ValueError("unknown metadata type")


def tensor_bytes(tensor):
    if tensor.device.type != "cpu":
        raise ValueError("CPU tensors required")
    # Checkpoints may contain column-major SVD cores. Canonicalize storage,
    # preserving logical element bits, shape and dtype (not physical strides).
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy()


def save_archive(value, directory, layout, chunk_bytes=1024 * 1024, level=3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if chunk_bytes < 8 or layout not in LAYOUTS:
        raise ValueError("invalid archive configuration")
    tree, tensors = flatten(value)
    manifest = {"schema": SCHEMA, "byteorder": sys.byteorder, "layout": layout,
                "zstd_level": level, "chunk_bytes": chunk_bytes, "tree": tree, "tensors": []}
    started = time.perf_counter()
    with (directory / "payload.bin").open("wb") as output:
        for name, tensor in tensors:
            raw = tensor_bytes(tensor)
            width = tensor.element_size()
            chunk_size = chunk_bytes // width * width
            entry = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                     "width": width, "numel": tensor.numel(), "chunks": []}
            source_hash = hashlib.sha256()
            for offset in range(0, len(raw), chunk_size):
                block = raw[offset:offset + chunk_size].tobytes()
                source_hash.update(block)
                encoded, compressed = encode_chunk(block, width, layout, level)
                entry["chunks"].append({"offset": output.tell(), "bytes": len(encoded), "raw_bytes": len(block),
                                        "compressed": compressed, "raw_sha256": digest_bytes(block)})
                output.write(encoded)
            entry["source_sha256"] = source_hash.hexdigest()
            manifest["tensors"].append(entry)
    manifest["payload_sha256"] = digest_file(directory / "payload.bin")
    write_json(directory / "manifest.json", manifest)
    encode_seconds = time.perf_counter() - started
    # Reopen the persisted archive, decode every chunk, and compare with source.
    started = time.perf_counter()
    verify_archive(directory, value)
    return {"directory": str(directory), "tensor_count": len(tensors), "bitwise_equal": True,
            "raw_bytes": sum(t.numel() * t.element_size() for _, t in tensors),
            "archive_bytes": sum((directory / n).stat().st_size for n in ("manifest.json", "payload.bin")),
            "encode_seconds": encode_seconds, "decode_verify_seconds": time.perf_counter() - started,
            "manifest_sha256": digest_file(directory / "manifest.json"), "payload_sha256": manifest["payload_sha256"]}


def read_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["byteorder"] != sys.byteorder:
        raise ValueError("incompatible archive schema or byte order")
    if digest_file(directory / "payload.bin") != manifest["payload_sha256"]:
        raise ValueError("archive payload hash mismatch")
    return manifest


def decoded_chunks(stream, entry, layout):
    h = hashlib.sha256()
    size = 0
    for chunk in entry["chunks"]:
        stream.seek(chunk["offset"])
        encoded = stream.read(chunk["bytes"])
        raw = decode_chunk(encoded, entry["width"], chunk["raw_bytes"] // entry["width"], layout, chunk["compressed"])
        if digest_bytes(raw) != chunk["raw_sha256"]:
            raise ValueError("decoded chunk hash mismatch")
        size += len(raw)
        h.update(raw)
        yield raw
    if size != entry["numel"] * entry["width"] or h.hexdigest() != entry["source_sha256"]:
        raise ValueError("decoded tensor hash or size mismatch")


def verify_archive(directory, reference=None):
    directory = Path(directory)
    manifest = read_manifest(directory)
    if reference is not None:
        tree, tensors = flatten(reference)
        if tree != manifest["tree"] or len(tensors) != len(manifest["tensors"]):
            raise ValueError("checkpoint metadata mismatch")
    with (directory / "payload.bin").open("rb") as stream:
        for i, entry in enumerate(manifest["tensors"]):
            if reference is not None:
                name, tensor = tensors[i]
                if name != entry["name"] or list(tensor.shape) != entry["shape"] or str(tensor.dtype) != entry["dtype"]:
                    raise ValueError("tensor metadata mismatch")
                source = tensor_bytes(tensor)
            offset = 0
            for raw in decoded_chunks(stream, entry, manifest["layout"]):
                if reference is not None and raw != source[offset:offset + len(raw)].tobytes():
                    raise ValueError("tensor bitwise round trip failed")
                offset += len(raw)
    return manifest


def load_archive(directory):
    """Decode a self-contained archive; no source checkpoint is required."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    tensors = []
    with (directory / "payload.bin").open("rb") as stream:
        for entry in manifest["tensors"]:
            dtype = getattr(torch, entry["dtype"].removeprefix("torch."))
            tensor = torch.empty(entry["shape"], dtype=dtype)
            if tensor.numel() != entry["numel"] or tensor.element_size() != entry["width"]:
                raise ValueError("invalid tensor metadata")
            destination = tensor_bytes(tensor)
            offset = 0
            for raw in decoded_chunks(stream, entry, manifest["layout"]):
                destination[offset:offset + len(raw)] = np.frombuffer(raw, np.uint8)
                offset += len(raw)
            tensors.append(tensor)
    return inflate(manifest["tree"], tensors)
