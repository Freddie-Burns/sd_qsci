import numpy as np

def spin_expand_1e(h1_spatial: np.ndarray) -> np.ndarray:
    """
    Expand spatial 1e integrals (n,n) into spin-orbital (2n,2n) in BLOCK order (all α then all β).
    """
    n = h1_spatial.shape[0]
    h = np.zeros((2*n, 2*n), dtype=h1_spatial.dtype)
    # αα block
    h[0:n, 0:n] = h1_spatial
    # ββ block
    h[n:2*n, n:2*n] = h1_spatial
    return h


def spin_expand_2e_phys(g_phys: np.ndarray) -> np.ndarray:
    """
    Expand spatial 2e integrals in physicist order (p,r,q,s) into spin-orbital (BLOCK α…β).
    Nonzero only when spin(P)==spin(R) and spin(Q)==spin(S).
    Returns G[P,Q,R,S] with shape (2n,2n,2n,2n).
    """
    n = g_phys.shape[0]
    G = np.zeros((2*n, 2*n, 2*n, 2*n), dtype=g_phys.dtype)

    # αα,αα
    G[0:n, 0:n, 0:n, 0:n] = g_phys
    # ββ,ββ
    G[n:2*n, n:2*n, n:2*n, n:2*n] = g_phys
    # αβ,αβ  (Coulomb α with α, β with β)
    G[0:n, n:2*n, 0:n, n:2*n] = g_phys
    # βα,βα
    G[n:2*n, 0:n, n:2*n, 0:n] = g_phys
    return G
