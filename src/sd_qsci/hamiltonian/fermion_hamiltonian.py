import numpy as np
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
        H += 0.5 * g2_phys[p, q, r, s] * (adag[p] @ adag[q] @ a[s] @ a[r])

    # nuclear repulsion shift
    if enuc != 0.0:
        H += enuc * identity(dim, format="csr", dtype=np.complex128)

    return H