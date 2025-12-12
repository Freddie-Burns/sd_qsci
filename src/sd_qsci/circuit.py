"""
Quantum-circuit helpers for orbital rotation workflows.

This module lifts the unitary-rotation circuit construction from the
notebook `notebooks/dev/01_verify_unitary_qc.ipynb` into reusable
functions. It focuses on:

- Preparing a Hartree–Fock Slater determinant in Jordan–Wigner mapping
- Applying an orbital rotation corresponding to RHF -> UHF unitaries
- Optionally optimizing the circuit for a single Slater determinant
- Running the circuit on a statevector simulator

Dependencies:
- ffsim (for JW preparation and orbital rotation circuit templates)
- qiskit and qiskit-aer (for circuit container and simulation)

These dependencies are optional at package import time and are imported
lazily inside functions. If they are missing, the functions will raise a
clear ImportError with installation hints.
"""

from __future__ import annotations  # prevents sphinx docs type error

import numpy as np
import ffsim
from line_profiler import profile
from pyscf.cc import CCSD
from pyscf.gto import Mole
from pyscf.scf.hf import RHF
from pyscf.scf.uhf import UHF
from pyscf import cc as pyscf_cc
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import Aer

from .utils import uhf_to_rhf_unitaries


@profile
def get_lucj_circuit(
    ccsd_obj: CCSD,
    backend,
    n_reps: int = 1,
    pairs_aa: list[tuple[int]] = None,
    pairs_ab: list[tuple[int]] = None,
    ab_coupling_interval: int = 1, # only used if pairs_ab not set
    homo_lumo_expansion: bool = True
) -> QuantumCircuit:
    """
    Create a local unitary cluster Jastrow ansatz circuit.
    """
    # Number of orbitals and electrons in the molecule object
    norb, nelec = int(ccsd_obj.mol.nao), ccsd_obj.mol.nelec

    # Initialize from ccsd t2 amplitudes (calculated with PySCF)
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(ccsd_obj.t2, n_reps=n_reps)
    if pairs_aa is None:
        # Setting up square "ladder" interaction graph, same-spin couplings on a line topology:
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    if pairs_ab is None:
        # Diagonal Coulomb interactions between up/down electrons sharing same spatial orbital
        pairs_ab = [(p,p) for p in range(0,norb,ab_coupling_interval)]
    interaction_pairs = (pairs_aa, pairs_ab)

    # Enforce locality constraints (i.e. to respect the interaction graph specified above):
    lparams = ucj_op.to_parameters(interaction_pairs=interaction_pairs)
    lucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
        lparams,
        norb=norb,
        n_reps=n_reps,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=False
    )

    # Create a QuantumCircuit, initialized from the HF state:
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)#; circuit.barrier()
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(lucj_op), qubits)
    circuit = circuit.decompose().decompose()

    if homo_lumo_expansion:
        # lightcone optimization
        n_gates = len(circuit.data)
        n_gates_prior = None
        while n_gates != n_gates_prior:
            n_gates_prior = n_gates
            drop_gate_indices = []
            for index,gate in enumerate(circuit.data):
                if gate.name == "xx_plus_yy":
                    gate_qubits = {q._index for q in gate.qubits}
                    gate_intersections = [gate_prior.name for gate_prior in circuit.data[:index] if gate_prior.name != 'barrier' and gate_qubits.intersection({q._index for q in gate_prior.qubits})!=set()]
                    if gate_intersections == [] or gate_intersections == ['x', 'x']:
                        drop_gate_indices.append(index)
            drop_gate_indices = list(sorted(set(drop_gate_indices)))
            for i,index in enumerate(drop_gate_indices):
                circuit.data.pop(index-i)
            n_gates = len(circuit.data)

    # reorder spin convention uuuu...dddd... -> udududud...
    qc = transpile(circuit, initial_layout={q:2*(i%norb)+(i//norb) for i,q in enumerate(qubits)}, optimization_level=0)
    qc = qc.copy() # Not sure why, don't question it I guess
    qc.measure_all()
    return transpile(qc, backend=backend, optimization_level=3)


@profile
def orbital_rotation_circuit(
    nao: int,
    nelec: tuple[int, int],
    Ua: np.ndarray,
    Ub: np.ndarray,
    *,
    prepare_hf: bool = True,
    optimize_single_slater: bool = True,
):
    """
    Create a Qiskit circuit that prepares a Hartree–Fock determinant and
    applies an orbital rotation given by (Ua, Ub) in Jordan–Wigner mapping.

    Parameters
    ----------
    nao : int
        Number of spatial orbitals (PySCF `mol.nao`). Total qubits = 2*n_orb.
    nelec : tuple[int, int]
        Number of alpha and beta electrons `(n_alpha, n_beta)`.
    Ua, Ub : np.ndarray
        Unitary matrices (shape `(n_orb, n_orb)`) mapping RHF MOs to UHF MOs
        for alpha and beta channels, respectively. In the original notebook,
        we used `(Ua.T, Ub.T)` inside the `OrbitalRotationJW` block. That
        convention is preserved here.
    prepare_hf : bool, default True
        If True, prepend a HF determinant preparation for `(n_orb, n_elec)`.
    optimize_single_slater : bool, default True
        If True, run `ffsim.qiskit.PRE_INIT` pass manager to simplify the
        rotation for a single Slater determinant input.

    Returns
    -------
    qiskit.QuantumCircuit
        The constructed circuit.
    """
    qubits = QuantumRegister(2 * nao, name="q")
    qc = QuantumCircuit(qubits)

    if prepare_hf:
        qc.append(ffsim.qiskit.PrepareHartreeFockJW(nao, nelec), qubits)

    # Todo: why does this work, does ffsim swap row/col convention?
    Ua, Ub = Ua.T, Ub.T
    qc.append(ffsim.qiskit.OrbitalRotationJW(nao, (Ua, Ub)), qubits)

    if optimize_single_slater:
        # Optimize for a single Slater determinant state
        qc = ffsim.qiskit.PRE_INIT.run(qc)

    return qc


def rhf_uhf_orbital_rotation_circuit(
    mol: Mole,
    rhf: RHF,
    uhf: UHF,
    *,
    optimize_single_slater: bool = True,
):
    """
    Convenience wrapper that:
    1) ensures a UHF reference (running it from RHF if not provided),
    2) computes RHF->UHF unitaries `(Ua, Ub)`,
    3) builds the corresponding orbital-rotation circuit.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule defining `nao` and spin-resolved electron counts `nelec`.
    rhf : pyscf.scf.hf.RHF
        Converged RHF mean-field.
    uhf : pyscf.scf.uhf.UHF, optional
        If not provided, it is generated from `rhf` via `uhf_from_rhf`.
    optimize_single_slater : bool, default True
        Run the PRE_INIT optimization suitable for a single Slater input.

    Returns
    -------
    circuit : qiskit.QuantumCircuit
    """
    Ua, Ub = uhf_to_rhf_unitaries(mol, rhf, uhf)
    qc = orbital_rotation_circuit(
        nao=mol.nao,
        nelec=mol.nelec,
        Ua=Ua,
        Ub=Ub,
        prepare_hf=True,
        optimize_single_slater=optimize_single_slater,
    )
    return qc


@profile
def simulate(qc, *, optimization_level: int=1):
    """
    Execute a circuit on Qiskit Aer statevector simulator and return the
    resulting statevector.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        Circuit to simulate.
    optimization_level : int, default 1
        Transpiler optimization level before running.

    Returns
    -------
    statevector : qiskit.quantum_info.Statevector
        The final statevector object (has `.data` ndarray and utility methods).
    """
    backend = Aer.get_backend("statevector_simulator")
    tqc = transpile(qc, backend, optimization_level=optimization_level)
    job = backend.run(tqc)
    result = job.result()
    return result.get_statevector()


__all__ = [
    "orbital_rotation_circuit",
    "rhf_uhf_orbital_rotation_circuit",
    "lucj_block_spin_circuit",
    "lucj_block_spin_circuit_from_ccsd",
    "simulate",
]
