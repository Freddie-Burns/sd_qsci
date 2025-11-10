import numpy as np
import pytest


# 1) BLOCK spin ordering consistency (cross-spin one-electron blocks must be zero)
def test_spin_block_cross_terms_zero():
    from sd_qsci.hamiltonian.spin_blocks import spin_expand_1e, spin_expand_2e_phys

    # Small distinct 1e and 2e tensors to make violations obvious
    h = np.array([[0.7, -0.2],
                  [-0.2, 0.3]], dtype=float)
    g = np.arange(2 ** 4, dtype=float).reshape(2, 2, 2, 2)  # distinct entries

    H = spin_expand_1e(h)
    G = spin_expand_2e_phys(g)

    n = h.shape[0]
    # αα and ββ one-electron blocks equal the spatial h
    assert np.allclose(H[0:n, 0:n], h)
    assert np.allclose(H[n:2*n, n:2*n], h)
    # cross-spin one-electron couplings vanish
    assert np.allclose(H[0:n, n:2*n], 0.0)
    assert np.allclose(H[n:2*n, 0:n], 0.0)

    # Two-electron spin structure (BLOCK ordering): representative checks
    # αα,αα; ββ,ββ; αβ,αβ; βα,βα blocks equal g
    assert np.allclose(G[0:n, 0:n, 0:n, 0:n], g)
    assert np.allclose(G[n:2*n, n:2*n, n:2*n, n:2*n], g)
    assert np.allclose(G[0:n, n:2*n, 0:n, n:2*n], g)
    assert np.allclose(G[n:2*n, 0:n, n:2*n, 0:n], g)
    # A mismatched-spin slot should be zero, e.g. αα,αβ
    assert np.allclose(G[0:n, 0:n, 0:n, n:2*n], 0.0)


# 2) PySCF glue: RHF energy must match expectation value on BLOCK-ordered HF determinant
@pytest.mark.slow
def test_hamiltonian_from_pyscf_rhf_energy_matches_h2_sto3g():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    import numpy as _np

    from sd_qsci.hamiltonian.pyscf_glue import hamiltonian_from_pyscf

    # Build minimal system and run RHF with explicit tolerances for determinism
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Ang")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    rhf = mf.run()

    H = hamiltonian_from_pyscf(mol, rhf).toarray()

    # BLOCK ordering HF determinant: occupy α0 and β0 at bit positions 0 and nmo
    nmo = rhf.mo_coeff.shape[1]
    ket_index = (1 << 0) | (1 << nmo)  # α0 + β0
    psi = _np.zeros(H.shape[0], dtype=_np.complex128)
    psi[ket_index] = 1.0

    energy = _np.real(psi.conj() @ (H @ psi))
    # Be explicit with tolerance due to PySCF minor version differences
    assert _np.isclose(energy, rhf.e_tot, atol=1e-6), f"{energy=} vs {rhf.e_tot=}"


# 3) Qiskit Aer determinism with fixed seed (avoid optional GPU paths)
@pytest.mark.qiskit
def test_qiskit_aer_determinism_with_seed():
    qiskit = pytest.importorskip("qiskit")
    aer = pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    # Simple 1-qubit circuit with known amplitudes: Ry then phase
    qc = QuantumCircuit(1)
    theta = 0.3
    phi = -0.7
    qc.ry(theta, 0)
    qc.p(phi, 0)
    qc.save_statevector()

    backend = AerSimulator(method="statevector", seed_simulator=1234)

    # Run twice to verify determinism under a fixed seed
    res1 = backend.run(qc, shots=None).result().data(0)["statevector"]
    res2 = backend.run(qc, shots=None).result().data(0)["statevector"]

    # Known expected amplitudes for |0>, |1>
    import numpy as _np
    expected = _np.array([
        _np.cos(theta/2.0),
        _np.exp(1j * phi) * _np.sin(theta/2.0),
    ], dtype=complex)

    assert _np.allclose(res1, expected, atol=1e-12)
    assert _np.allclose(res2, expected, atol=1e-12)
    assert _np.allclose(res1, res2, atol=1e-15)
