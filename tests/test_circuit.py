import numpy as np
import pytest

from pyscf import gto, scf
from pyscf.scf.uhf import UHF

from sd_qsci.circuit import (
    orbital_rotation_circuit,
    rhf_uhf_orbital_rotation_circuit,
    simulate,
)


def test_orbital_rotation_circuit():
    # Optional deps for this test
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")

    nao = 2
    nelec = (1, 1)
    Ua = np.eye(nao)
    Ub = np.eye(nao)

    qc = orbital_rotation_circuit(
        nao=nao,
        nelec=nelec,
        Ua=Ua,
        Ub=Ub,
        prepare_hf=True,
        optimize_single_slater=True,
    )

    # Total qubits should be 2*n_orb (BLOCK spin ordering)
    assert qc.num_qubits == 2 * nao
    # Circuit should contain at least the HF preparation
    assert len(qc.data) > 0


def test_simulate_circuit_one_qubit_x():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.x(0)  # |0> -> |1>

    sv = simulate(qc)
    # Probability of |1> should be 1
    assert np.isclose(abs(sv.data[1]) ** 2, 1.0)


def test_rhf_uhf_orbital_rotation_circuit_smoke():
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")

    # Minimal H2 system (fast and deterministic)
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

    rhf = scf.RHF(mol)
    guess = rhf.get_init_guess(mol=mol, key="hcore")
    rhf.kernel(dm0=guess)

    qc, uhf, (Ua, Ub) = rhf_uhf_orbital_rotation_circuit(
        mol, rhf, optimize_single_slater=True
    )

    # UHF object returned
    assert isinstance(uhf, UHF)

    # Unitary shapes and circuit size
    n_orb = mol.nao
    assert Ua.shape == (n_orb, n_orb)
    assert Ub.shape == (n_orb, n_orb)
    assert qc.num_qubits == 2 * n_orb


# Todo: try odd number of atoms
# Todo: try different basis sets
# Todo: try non-linear geometries
@pytest.mark.parametrize(
    "n_atoms, spacing, basis", [
    (2, 0.7, "sto-3g"),  # H2 at equilibrium
    (2, 1.5, "sto-3g"),  # H2 stretched
    (4, 1.8, "sto-3g"),  # H4 moderate spacing
    (4, 3.0, "sto-3g"),  # H4 stretched (higher correlation)
    (6, 1.5, "sto-3g"),  # H6 chain
])
def test_uhf_orbital_rotation_energy_preservation(n_atoms, spacing, basis):
    """
    Test complete workflow for various hydrogen chain geometries:
    1. Create hydrogen chain molecule
    2. Calculate RHF and UHF solutions
    3. Find unitaries for UHF to RHF orbital rotation
    4. Build quantum circuit to apply unitary to RHF statevector
    5. Simulate the quantum circuit
    6. Create occupation number vector Hamiltonian in RHF basis
    7. Verify statevector energy equals original UHF energy
    """
    from scipy import sparse
    from sd_qsci.utils import uhf_from_rhf, uhf_to_rhf_unitaries
    from sd_qsci.hamiltonian import hamiltonian_from_pyscf

    # Step 1: Create hydrogen chain molecule with varying geometry
    coords = [(i * spacing, 0, 0) for i in range(n_atoms)]
    geometry = '; '.join([f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in coords])
    mol = gto.Mole()
    mol.build(
        atom=geometry,
        unit='Angstrom',
        basis=basis,
        charge=0,
        spin=0,
        verbose=0,
    )

    # Step 2: Calculate RHF and UHF solutions
    rhf = scf.RHF(mol)
    rhf.run()
    uhf = uhf_from_rhf(mol, rhf)

    # Store original UHF energy for comparison
    uhf_energy = uhf.e_tot

    # Step 3: Find unitaries for UHF to RHF orbital rotation
    Ua, Ub = uhf_to_rhf_unitaries(mol, rhf, uhf)

    # Verify unitaries are actually unitary matrices
    assert np.allclose(Ua @ Ua.conj().T, np.eye(mol.nao), atol=1e-10)
    assert np.allclose(Ub @ Ub.conj().T, np.eye(mol.nao), atol=1e-10)

    # Step 4: Build quantum circuit to apply unitary to RHF statevector
    qc = orbital_rotation_circuit(
        nao=mol.nao,
        nelec=mol.nelec,
        Ua=Ua,
        Ub=Ub,
        prepare_hf=True,
        optimize_single_slater=True,
    )

    # Verify circuit construction
    assert qc.num_qubits == 2 * mol.nao
    assert len(qc.data) > 0

    # Step 5: Simulate the quantum circuit
    statevector = simulate(qc)

    # Verify statevector is normalized
    assert np.isclose(np.linalg.norm(statevector.data), 1.0, atol=1e-10)

    # Step 6: Create occupation number vector Hamiltonian in RHF basis
    H = hamiltonian_from_pyscf(mol, rhf)  # scipy sparse matrix
    assert sparse.linalg.norm(H - H.getH()) < 1e-10

    # Step 7: Verify statevector energy equals original UHF energy
    sv_energy = (statevector.data.conj().T @ H @ statevector.data).real

    # The statevector energy should match the UHF energy to high precision
    # Since the circuit implements the exact orbital rotation transformation
    assert np.isclose(sv_energy, uhf_energy, atol=1e-8), (
        f"Statevector energy {sv_energy:.10f} does not match "
        f"UHF energy {uhf_energy:.10f} (diff: {abs(sv_energy - uhf_energy):.2e})"
    )

    # Additional check: verify energy is lower than RHF for stretched systems
    # (when UHF symmetry breaking is significant)
    rhf_energy = rhf.e_tot
    energy_diff = abs(uhf_energy - rhf_energy)

    # Print summary for debugging
    print(f"\n=== Energy Comparison (H{n_atoms}, spacing={spacing:.2f}Å, {basis}) ===")
    print(f"RHF energy: {rhf_energy:.10f}")
    print(f"UHF energy: {uhf_energy:.10f}")
    print(f"Statevector energy: {sv_energy:.10f}")
    print(f"RHF-UHF diff: {energy_diff:.2e}")
    print(f"UHF-SV diff: {abs(uhf_energy - sv_energy):.2e}")

    # For systems with significant UHF symmetry breaking, verify UHF < RHF
    if energy_diff > 1e-6:
        assert sv_energy < rhf_energy, (
            f"Statevector energy {sv_energy:.10f} should be lower than "
            f"RHF energy {rhf_energy:.10f} for this correlated system"
        )

