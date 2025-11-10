import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse import csc_matrix

# ----- Pauli / ladder matrices (sparse) -----
I2  = csr_matrix(np.eye(2))
Z   = csr_matrix(np.array([[1, 0], [0,-1]], dtype=complex))
Sp  = csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))   # sigma^+
Sm  = csr_matrix(np.array([[0, 0], [1, 0]], dtype=complex))   # sigma^-

def jw_creation(n_modes: int, j: int) -> csc_matrix:
    """
    Jordan–Wigner creation operator a_j^\dagger as a 2^n x 2^n sparse matrix.
    Mode index j \in [0, n_modes-1], with mode 0 the leftmost qubit.
    """
    op = None
    for k in range(n_modes):
        fac = Z if k < j else (Sp if k == j else I2)
        op = fac if op is None else kron(op, fac, format="csr")
    return op.tocsc()

def jw_annihilation(n_modes: int, j: int) -> csc_matrix:
    """
    Jordan–Wigner annihilation operator a_j as a 2^n x 2^n sparse matrix.
    """
    op = None
    for k in range(n_modes):
        fac = Z if k < j else (Sm if k == j else I2)
        op = fac if op is None else kron(op, fac, format="csr")
    return op.tocsc()

def number_op(n_modes: int, j: int) -> csc_matrix:
    """
    Number operator n_j = a_j^\dagger a_j.
    """
    a_dag = jw_creation(n_modes, j)
    a     = jw_annihilation(n_modes, j)
    return (a_dag @ a).tocsc()

# ---------- Total spin operators ----------
def spin_ops(n_spatial_orbs: int):
    """
    Build S_z, S_+, S_- for RHF ordering: [0..M-1]=alpha, [M..2M-1]=beta.
    Returns (Sz, Splus, Sminus) as sparse CSC matrices.
    """
    M = n_spatial_orbs
    n_modes = 2*M
    dim = 1 << n_modes

    # Pre-build number operators for efficiency
    n_alpha = [number_op(n_modes, p)       for p in range(M)]
    n_beta  = [number_op(n_modes, M + p)   for p in range(M)]

    # S_z
    Sz = sum((n_alpha[p] - n_beta[p]) for p in range(M)) * 0.5

    # S_+ = sum_p a^\dagger_{pα} a_{pβ}
    # S_- = sum_p a^\dagger_{pβ} a_{pα}
    Splus_terms  = []
    Sminus_terms = []
    for p in range(M):
        a_alpha_dag = jw_creation(n_modes, p)
        a_beta      = jw_annihilation(n_modes, M + p)
        a_beta_dag  = jw_creation(n_modes, M + p)
        a_alpha     = jw_annihilation(n_modes, p)

        Splus_terms.append((a_alpha_dag @ a_beta).tocsc())
        Sminus_terms.append((a_beta_dag @ a_alpha).tocsc())

    Splus  = sum(Splus_terms)
    Sminus = sum(Sminus_terms)

    return Sz.tocsc(), Splus.tocsc(), Sminus.tocsc()

def total_spin_S2(n_spatial_orbs: int) -> csc_matrix:
    """
    Construct the total spin operator S^2 as a sparse matrix.
    """
    Sz, Splus, Sminus = spin_ops(n_spatial_orbs)
    return (Sz @ Sz + 0.5 * (Splus @ Sminus + Sminus @ Splus)).tocsc()

# ---------- Utilities ----------
def expectation(op: csc_matrix, psi: np.ndarray) -> complex:
    """
    <psi| op |psi> for a statevector psi (1D numpy array). psi need not be normalized.
    """
    psi = psi.astype(complex)
    norm = np.vdot(psi, psi)
    if norm == 0:
        raise ValueError("State vector has zero norm.")
    psi_norm = psi / np.sqrt(norm)
    return np.vdot(psi_norm, op @ psi_norm)

# ---------- Example usage ----------
if __name__ == "__main__":
    # One spatial orbital => modes: [alpha0, beta0]
    M = 1
    S2 = total_spin_S2(M)

    # Your example state: (|00> + |10>)/sqrt(2)
    # Basis order is [|00>, |01>, |10>, |11>] with alpha first, then beta.
    psi = np.array([0.707, 0.0, 0.707, 0.0], dtype=float)

    val = expectation(S2, psi)
    print(f"<S^2> = {val.real:.6f}")

    # Some sanity checks:
    # |00>  (vacuum)   -> <S^2> = 0
    # |10>  (α only)   -> <S^2> = 3/4
    # |01>  (β only)   -> <S^2> = 3/4
    # |11>  (αβ pair)  -> <S^2> = 0  (singlet on one spatial orbital)
    for label, vec in {
        "|00>": np.array([1,0,0,0], float),
        "|10>": np.array([0,0,1,0], float),
        "|01>": np.array([0,1,0,0], float),
        "|11>": np.array([0,0,0,1], float),
    }.items():
        print(label, "->", expectation(S2, vec).real)
