"""Numerical parity between an output and a reference."""

import numpy as np


def compare(out, ref) -> dict:
    a = np.asarray(out, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    d = a - r
    finite = np.isfinite(a)
    return {
        "max_abs": float(np.abs(d).max()),
        "mean_abs": float(np.abs(d).mean()),
        "rel_l2": float(np.linalg.norm(d) / max(np.linalg.norm(r), 1e-12)),
        "cosine": float(a @ r / max(np.linalg.norm(a) * np.linalg.norm(r), 1e-12)),
        "nonfinite": int((~finite).sum()),
    }
