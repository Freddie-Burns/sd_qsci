import numpy as np
from pyscf import fci
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import Statevector


def fci_energy(rhf):
    """
    Calculate Full Configuration Interaction (FCI) energy.

    Parameters
    ----------
    rhf : scf.RHF
        Converged RHF calculation object.

    Returns
    -------
    float
        FCI ground state energy in Hartree.
    """
    ci_solver = fci.FCI(rhf)
    fci_energy, fci_vec = ci_solver.kernel()
    return fci_energy


def qsci_energy(H: csr_matrix, statevector: Statevector):
    """
    Calculate Quantum Subspace Configuration Interaction (QSCI) energy.

    Extracts the significant configurations from the statevector (based on
    amplitude threshold), constructs the Hamiltonian in this reduced subspace,
    and solves for the ground state energy.

    Parameters
    ----------
    H : scipy.sparse matrix
        Full Hamiltonian matrix in the computational basis (Fock space).
    statevector : circuit.Statevector
        Quantum statevector with amplitudes for all basis configurations.

    Returns
    -------
    E0 : float
        QSCI ground state energy in Hartree.
    idx : np.ndarray
        Array of configuration indices used in the QSCI subspace.

    Notes
    -----
    - Configurations with |amplitude| < 1e-12 are filtered out
    - For small subspaces (≤2 dimensions), uses dense eigenvalue solver
    - For larger subspaces, uses sparse eigenvalue solver (eigsh)
    """
    idx = np.argwhere(np.abs(statevector.data) > 1e-12).ravel()
    H_sub = H[np.ix_(idx, idx)]

    # Handle small matrices where eigsh would fail
    if H_sub.shape[0] <= 2:
        eigenvalues, eigenvectors = eigh(H_sub.toarray())
        E0 = eigenvalues[0]
    else:
        E0, psi0 = eigsh(H_sub, k=1, which='SA')
        E0 = E0[0]

    return E0, idx