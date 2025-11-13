import numpy as np
from line_profiler import profile
from scipy.sparse import coo_matrix, csr_matrix, identity
from .fermion_ops import annihilate


def ladder_operators(N):
    """Generate annihilation and creation operator matrices for N spin orbitals.

    Parameters
    ----------
    N : int
        Number of spin orbitals.

    Returns
    -------
    tuple of lists
        ([a_p], [a_p†]) where each element is a sparse matrix representing
        the annihilation and creation operators respectively.
    """
    dim = 1 << N
    a_ops = []
    adag_ops = []
    for p in range(N):
        rows, cols, data = [], [], []
        for ket in range(dim):
            res = annihilate(ket, p)
            if res:
                ph, bra = res
                rows.append(bra)
                cols.append(ket)
                data.append(ph)
        A = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128).tocsr()
        a_ops.append(A)
        adag_ops.append(A.conjugate().transpose())
    return a_ops, adag_ops


@profile
def hamiltonian_matrix(h1, g2_phys, enuc=0.0):
    """Build the full Fock-space Hamiltonian using precomputed sparse ladder operators.

    Parameters
    ----------
    h1 : ndarray
        One-electron integrals matrix of shape (N, N).
    g2_phys : ndarray
        Two-electron integrals tensor in physicist notation of shape (N, N, N, N).
    enuc : float, optional
        Nuclear repulsion energy, by default 0.0.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix representation of the many-body Hamiltonian.
    """
    N = h1.shape[0]
    a, adag = ladder_operators(N)
    dim = 1 << N
    H = csr_matrix((dim, dim), dtype=np.complex128)

    # 1-electron term
    for p, q in np.argwhere(abs(h1) > 1e-16):
        H += h1[p, q] * adag[p] @ a[q]

    # 2-electron term
    for p, q, r, s in np.argwhere(abs(g2_phys) > 1e-16):
        g2 = g2_phys[p, q, r, s]
        op = adag[p] @ adag[q] @ a[s] @ a[r]
        H += 0.5 * g2 * op
        # H += 0.5 * g2_phys[p, q, r, s] * (adag[p] @ adag[q] @ a[s] @ a[r])

    # nuclear repulsion shift
    if enuc != 0.0:
        H += enuc * identity(dim, format="csr", dtype=np.complex128)

    return H


@profile
def hamiltonian_matrix_direct(h1, g2_phys, enuc=0.0):
    """Build Hamiltonian by directly computing matrix elements between basis states.

    Parameters
    ----------
    h1 : ndarray
        One-electron integrals matrix of shape (N, N).
    g2_phys : ndarray
        Two-electron integrals tensor in physicist notation of shape (N, N, N, N).
    enuc : float, optional
        Nuclear repulsion energy, by default 0.0.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix representation of the many-body Hamiltonian.
    """
    N = h1.shape[0]
    dim = 1 << N

    rows, cols, data = [], [], []

    # Iterate over all basis states (kets)
    for ket in range(dim):
        # Diagonal: nuclear repulsion
        if enuc != 0.0:
            rows.append(ket)
            cols.append(ket)
            data.append(enuc)

        # One-electron terms: <bra| a†_p a_q |ket>
        for p, q in np.argwhere(abs(h1) > 1e-16):
            # Check if orbital q is occupied in ket
            if not (ket & (1 << q)):
                continue
            # Check if orbital p is empty in ket
            if ket & (1 << p):
                continue

            # Compute phase and resulting bra state
            bra = ket ^ (1 << q) ^ (1 << p)
            phase = 1

            # Phase from annihilating q
            phase *= (-1) ** bin(ket & ((1 << q) - 1)).count('1')
            # Phase from creating p
            intermediate = ket ^ (1 << q)
            phase *= (-1) ** bin(intermediate & ((1 << p) - 1)).count('1')

            rows.append(bra)
            cols.append(ket)
            data.append(phase * h1[p, q])

        # Two-electron terms: <bra| a†_p a†_q a_s a_r |ket>
        for p, q, r, s in np.argwhere(abs(g2_phys) > 1e-16):
            # Check occupancy requirements
            if not (ket & (1 << r)):
                continue
            if not (ket & (1 << s)):
                continue
            if ket & (1 << p):
                continue
            if ket & (1 << q):
                continue

            # Cannot annihilate same orbital twice
            if r == s:
                continue
            # Cannot create same orbital twice
            if p == q:
                continue

            # Apply operators: a_r, then a_s, then a†_q, then a†_p
            state = ket
            phase = 1

            # Annihilate r
            phase *= (-1) ** bin(state & ((1 << r) - 1)).count('1')
            state ^= (1 << r)

            # Annihilate s
            phase *= (-1) ** bin(state & ((1 << s) - 1)).count('1')
            state ^= (1 << s)

            # Create q
            phase *= (-1) ** bin(state & ((1 << q) - 1)).count('1')
            state ^= (1 << q)

            # Create p
            phase *= (-1) ** bin(state & ((1 << p) - 1)).count('1')
            bra = state ^ (1 << p)

            rows.append(bra)
            cols.append(ket)
            data.append(0.5 * phase * g2_phys[p, q, r, s])

    # Build sparse matrix
    H = coo_matrix((data, (rows, cols)), shape=(dim, dim),
                   dtype=np.complex128).tocsr()

    return H