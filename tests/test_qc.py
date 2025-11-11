import numpy as np
import pytest

from pyscf import gto, scf
from pyscf.scf.uhf import UHF

from sd_qsci.qc import (
    build_orbital_rotation_circuit,
    rhf_uhf_orbital_rotation_circuit,
    run_statevector,
)


def test_build_orbital_rotation_circuit_structure():
    # Optional deps for this test
    pytest.importorskip("ffsim")
    pytest.importorskip("qiskit")

    n_orb = 2
    n_elec = (1, 1)
    Ua = np.eye(n_orb)
    Ub = np.eye(n_orb)

    qc = build_orbital_rotation_circuit(
        n_orb=n_orb,
        n_elec=n_elec,
        Ua=Ua,
        Ub=Ub,
        prepare_hf=True,
        optimize_single_slater=True,
    )

    # Total qubits should be 2*n_orb (BLOCK spin ordering)
    assert qc.num_qubits == 2 * n_orb
    # Circuit should contain at least the HF preparation
    assert len(qc.data) > 0


def test_run_statevector_one_qubit_x():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.x(0)  # |0> -> |1>

    sv = run_statevector(qc)
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
