"""A flat binary file of named tensors with a JSON header, for runtimes that cannot read npz or pickle.

Layout (all integers little-endian):

    offset 0       8 bytes   magic, ASCII "SVLABNDL"
    offset 8       8 bytes   uint64 H: length of the header in bytes
    offset 16      H bytes   UTF-8 JSON header, padded with spaces so that 16 + H is a multiple of 64
    offset 16 + H            data section

The header is ``{"format": "smolvla-split-bundle", "version": 1, "meta": {...}, "tensors": [...]}``. Each
tensor entry is ``{"name", "dtype", "shape", "offset", "nbytes"}``: ``offset`` counts from the start of the
data section and is a multiple of 64, and the bytes are the tensor in C (row-major) order, little-endian.
dtypes are float32, float16, int64 and bool (one byte per element, 0 or 1). docs/SPLIT_GRAPH.md lists the
tensors tools/dump_bundle.py writes.
"""

import json
import math
import struct
from pathlib import Path

import numpy as np

MAGIC = b"SVLABNDL"
FORMAT = "smolvla-split-bundle"
VERSION = 1
ALIGN = 64
DTYPES = {"float32": np.float32, "float16": np.float16, "int64": np.int64, "bool": np.bool_}


def _aligned(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


def write_bundle(path: Path, tensors: dict[str, np.ndarray], meta: dict) -> None:
    entries, blobs, end = [], [], 0
    for name, array in tensors.items():
        array = np.asarray(array)
        if array.dtype.name not in DTYPES:
            raise TypeError(f"{name}: dtype {array.dtype.name} is not one of {list(DTYPES)}")
        blob = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<")).tobytes()
        offset = _aligned(end)
        entries.append({"name": name, "dtype": array.dtype.name, "shape": list(array.shape), "offset": offset,
                        "nbytes": len(blob)})
        blobs.append(blob)
        end = offset + len(blob)
    header = json.dumps({"format": FORMAT, "version": VERSION, "meta": meta, "tensors": entries}).encode()
    header += b" " * (_aligned(16 + len(header)) - 16 - len(header))
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<Q", len(header)) + header)
        written = 0
        for entry, blob in zip(entries, blobs, strict=True):
            f.write(b"\0" * (entry["offset"] - written) + blob)
            written = entry["offset"] + len(blob)


MAX_HEADER_BYTES = 1 << 30


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_number(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def read_bundle(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Every tensor of the bundle at ``path``, and its meta. Raises ValueError naming the file and the problem.

    It accepts the files the C++ reader (cpp/src/bundle.cpp) accepts, apart from meta.config, which the C++ reader
    also checks against the contract. Sizes are checked in Python integers, so no shape can overflow into a size
    that looks valid.
    """
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f"{path} is not a bundle (magic {raw[:8]!r})")
    if len(raw) < 16:
        raise ValueError(f"{path} is too short to be a bundle")
    (size,) = struct.unpack_from("<Q", raw, 8)
    data = 16 + size
    if size > MAX_HEADER_BYTES or data > len(raw):
        raise ValueError(f"{path}: header length {size} out of range")
    if data % ALIGN:
        raise ValueError(f"{path}: the data section, at byte {data}, is not 64-byte aligned")
    try:
        header = json.loads(raw[16:data])
    except ValueError as e:  # including invalid UTF-8
        raise ValueError(f"{path}: invalid header JSON: {e}") from e
    if not (isinstance(header, dict) and header.get("format") == FORMAT and type(header.get("version")) is int
            and header["version"] == VERSION):
        raise ValueError(f"{path}: unsupported bundle format (expected {FORMAT} version {VERSION})")
    for key, kind in (("meta", dict), ("tensors", list)):
        if not isinstance(header.get(key), kind):
            raise ValueError(f'{path}: "{key}" must be a JSON {"object" if kind is dict else "array"}')
    meta = header["meta"]
    if "euler" in meta:
        euler = meta["euler"]
        if not (isinstance(euler, dict) and _is_number(euler.get("dt")) and isinstance(euler.get("t"), list)
                and all(_is_number(t) for t in euler["t"])):
            raise ValueError(f'{path}: meta.euler must be {{"dt": number, "t": [number, ...]}}')

    tensors = {}
    for i, e in enumerate(header["tensors"]):
        name = e.get("name") if isinstance(e, dict) else None
        if not isinstance(name, str) or not {"dtype", "shape", "offset", "nbytes"} <= e.keys():
            raise ValueError(f"{path}: tensor entry {i} needs name, dtype, shape, offset and nbytes: {e}")
        if name in tensors:
            raise ValueError(f"{path}: duplicate tensor {name}")
        shape = e["shape"]
        if not isinstance(e["dtype"], str) or e["dtype"] not in DTYPES:
            raise ValueError(f"{path}: {name} has dtype {e['dtype']}, not one of {list(DTYPES)}")
        if not isinstance(shape, list) or not all(_is_count(d) for d in shape):
            raise ValueError(f"{path}: {name} has shape {shape}; dimensions must be non-negative integers")
        dtype = np.dtype(DTYPES[e["dtype"]]).newbyteorder("<")
        nbytes = math.prod(shape) * dtype.itemsize
        if not _is_count(e["nbytes"]) or nbytes != e["nbytes"]:
            raise ValueError(f"{path}: {name} has {e['nbytes']} bytes for shape {shape} {e['dtype']}, "
                             f"which needs {nbytes}")
        offset = e["offset"]
        if not _is_count(offset) or offset % ALIGN or data + offset + nbytes > len(raw):
            raise ValueError(f"{path}: {name} at offset {offset} is not 64-byte aligned or runs past the end of "
                             "the file")
        array = np.frombuffer(raw, dtype=dtype, count=nbytes // dtype.itemsize, offset=data + offset).reshape(shape)
        tensors[name] = array.astype(dtype.newbyteorder("="))
    return tensors, meta
