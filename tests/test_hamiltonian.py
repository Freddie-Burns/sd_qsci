import numpy as np
from pyscf import gto, scf
from sd_qsci.hamiltonian.spin_blocks import spin_expand_1e, spin_expand_2e_phys
from sd_qsci.hamiltonian.fermion_ops import create, annihilate
from sd_qsci.hamiltonian.fermion_hamiltonian import hamiltonian_matrix
from sd_qsci.hamiltonian.pyscf_glue import occ_hamiltonian_from_pyscf
from sd_qsci.hamiltonian.checks import rhf_energy_from_mo_integrals

def test_create_annihilate_roundtrip():
    # Start with |0101> over 4 modes => state integer with bits at {0,2}
    state = (1 << 0) | (1 << 2)

    # a_0 removes occupation at 0
    res = annihilate(state, 0); assert res is not None
    ph1, s1 = res
    # then a_0^† brings us back
    res2 = create(s1, 0); assert res2 is not None
    ph2, s2 = res2

    assert s2 == state
    # total phase squared must be +1 for round-trip on same site
    assert ph1 * ph2 in (+1, -1)

def test_annihilate_on_empty_is_none():
    state = 0  # |0000>
    assert annihilate(state, 1) is None

def test_create_on_filled_is_none():
    state = (1 << 1)  # |0010> -> bit 1 set
    assert create(state, 1) is None

def test_spin_expand_1e_blocks():
    h = np.array([[1.0, 0.2],[0.2, 0.5]])
    H = spin_expand_1e(h)
    n = h.shape[0]
    # αα and ββ match h; cross-spin are zero
    assert np.allclose(H[0:n, 0:n], h)
    assert np.allclose(H[n:2*n, n:2*n], h)
    assert np.allclose(H[0:n, n:2*n], 0.0)
    assert np.allclose(H[n:2*n, 0:n], 0.0)

def test_spin_expand_2e_phys_blocks():
    n = 2
    g = np.arange(n ** 4, dtype=float).reshape(n, n, n, n)  # distinct entries
    G = spin_expand_2e_phys(g)

    # αα,αα block equals g
    assert np.allclose(G[0:n, 0:n, 0:n, 0:n], g)
    # ββ,ββ block equals g
    assert np.allclose(G[n:2 * n, n:2 * n, n:2 * n, n:2 * n], g)
    # αβ,αβ block equals g
    assert np.allclose(G[0:n, n:2 * n, 0:n, n:2 * n], g)
    # βα,βα block equals g
    assert np.allclose(G[n:2 * n, 0:n, n:2 * n, 0:n], g)
    # A mismatched-spin slot should be zero, e.g. αα,αβ
    assert np.allclose(G[0:n, 0:n, 0:n, n:2 * n], 0.0)

def test_fermionic_h_one_body_coupling():
    # N=2 spin-orbitals; one-body hop between 0 and 1
    h1 = np.array([[0.0, 1.0],
                   [1.0, 0.0]])
    g2 = np.zeros((2,2,2,2))  # no 2e
    H = hamiltonian_matrix(h1, g2, enuc=0.0).toarray()

    # |01> (state=2) coupled to |10> (state=1) with +1 (up to fermionic sign)
    # Basis ordering is integer index of bitstring: |b1 b0>, LSB is orbital 0.
    ket_01 = 1 << 1  # occupies mode 1
    ket_10 = 1 << 0  # occupies mode 0
    assert abs(H[ket_10, ket_01]) == 1.0
    assert abs(H[ket_01, ket_10]) == 1.0

def test_h2_sto3g_rhf_energy_matches():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.0", basis="sto-3g")
    rhf = scf.RHF(mol).run()

    H = occ_hamiltonian_from_pyscf(mol, rhf).toarray()

    # BLOCK ordering HF determinant: occupy α0 and β0 -> bit positions 0 and nmo
    nmo = rhf.mo_coeff.shape[1]
    ket_index = (1 << 0) | (1 << nmo)  # α0 + β0
    psi = np.zeros(H.shape[0], dtype=np.complex128); psi[ket_index] = 1.0

    energy = np.real(psi.conj() @ (H @ psi))
    assert np.isclose(energy, rhf.e_tot, atol=1e-6), f"{energy=} vs {rhf.e_tot=}"

def test_rhf_energy_matches_pyscf_h2():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.0", basis="sto-3g")
    rhf = scf.RHF(mol).run()

    e_from_ints = rhf_energy_from_mo_integrals(mol, rhf)
    assert np.isclose(e_from_ints, rhf.e_tot, atol=1e-8), (
        f"{e_from_ints=} vs {rhf.e_tot=}"
    )

def test_rhf_energy_matches_pyscf_h4():
    # Use geometry without symmetry
    geometry = f"H 0 0 0; H 1 0 1; H 0 2 3; H 0 -1 4.5"
    mol = gto.M(atom=geometry, basis="sto-3g")
    rhf = scf.RHF(mol).run()

    e_from_ints = rhf_energy_from_mo_integrals(mol, rhf)
    assert np.isclose(e_from_ints, rhf.e_tot, atol=1e-8), (
        f"{e_from_ints=} vs {rhf.e_tot=}"
    )
