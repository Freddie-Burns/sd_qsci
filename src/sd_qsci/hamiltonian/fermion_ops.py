"""
Fermionic operators for computational basis states.

This module provides fundamental fermionic creation and annihilation operators
that act on computational basis states represented as integers. The operators
properly account for fermionic anticommutation relations by tracking phase
factors based on the occupation of lower-indexed modes.

Functions
---------
annihilate(state, q)
    Apply fermionic annihilation operator a_q to a basis state.
create(state, p)
    Apply fermionic creation operator a_p† to a basis state.
"""

def _is_occupied(state: int, i: int) -> bool:
    return (state >> i) & 1

def _phase_for(site: int, state: int) -> int:
    """Return (-1)^(# of occupied modes with index < site)."""
    mask = (1 << site) - 1
    return -1 if ((state & mask).bit_count() % 2) else 1

def annihilate(state: int, q: int):
    """
    Apply the fermionic annihilation operator ``a_q`` to the computational
    basis state ``|state>``.

    Returns
    -------
    (phase, new_state) or ``None``
        Returns ``None`` if the occupation at mode ``q`` is zero.
    """
    if not _is_occupied(state, q):
        return None
    phase = _phase_for(q, state)
    return phase, (state & ~(1 << q))

def create(state: int, p: int):
    """
    Apply the fermionic creation operator ``a_p†`` to the computational
    basis state ``|state>``.

    Returns
    -------
    (phase, new_state) or ``None``
        Returns ``None`` if the occupation at mode ``p`` is already one.
    """
    if _is_occupied(state, p):
        return None
    phase = _phase_for(p, state)
    return phase, (state | (1 << p))
