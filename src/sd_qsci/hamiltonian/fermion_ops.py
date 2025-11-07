# file: fermion_ops.py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def _popcnt(x: int) -> int:
    return x.bit_count()

def _is_occupied(state: int, i: int) -> bool:
    return (state >> i) & 1

def _phase_for(site: int, state: int) -> int:
    """(-1)^(# of occupied modes with index < site)."""
    mask = (1 << site) - 1
    return -1 if (_popcnt(state & mask) % 2) else 1

def annihilate(state: int, q: int):
    """Apply a_q to |state>. Returns (phase, new_state) or None if occupation is 0."""
    if not _is_occupied(state, q):
        return None
    phase = _phase_for(q, state)
    return phase, (state & ~(1 << q))

def create(state: int, p: int):
    """Apply a_p^\dagger to |state>. Returns (phase, new_state) or None if occupation is 1."""
    if _is_occupied(state, p):
        return None
    phase = _phase_for(p, state)
    return phase, (state | (1 << p))
