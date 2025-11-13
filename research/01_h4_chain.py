"""
Create H4 chain and compare energies and spins from RHF, UHF, FCI, and QSCI.
Ensure the statevector after orbital rotation gives the same energy and spin
as the input UHF state.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, scf, fci
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from sd_qsci.utils import uhf_from_rhf, uhf_to_rhf_unitaries
from sd_qsci import circuit, hamiltonian, spin, utils


def main():
    """
    Run H4 chain energy calculations across multiple bond lengths.

    Performs RHF, UHF, FCI, and QSCI calculations on a linear H4 chain
    for bond lengths ranging from 0.5 to 3.0 Angstrom. Generates a plot
    comparing the energy methods and saves it to 'figures/h4_chain_energies.png'.

    The workflow for each bond length:
    1. Build H4 chain molecule
    2. Run RHF calculation
    3. Generate UHF from RHF (symmetry-breaking)
    4. Create quantum circuit for orbital rotation (RHF → UHF)
    5. Simulate circuit to obtain statevector
    6. Calculate FCI energy (exact benchmark)
    7. Calculate QSCI energy from statevector subspace
    8. Store results for plotting

    Notes
    -----
    Asserts that the statevector energy matches the UHF energy to validate
    the orbital rotation circuit implementation.
    """
    rhf_energies = []
    uhf_energies = []
    fci_energies = []
    qsci_energies = []
    bond_lengths = []

    for bond_length in np.linspace(0.5, 3, 2):
        print(f"Running bond length: {bond_length:.2f} Angstrom")
        mol = build_h4_chain(bond_length)
        rhf = scf.RHF(mol).run()
        uhf = uhf_from_rhf(mol, rhf)
        qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
        sv = circuit.simulate(qc)
        H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)
        sv_energy = (sv.data.conj().T @ H @ sv.data).real
        assert np.isclose(sv_energy, uhf.e_tot)

        fci_energy = calc_fci_energy(rhf)
        qsci_energy = calc_qsci_energy(H, sv)

        bond_lengths.append(bond_length)
        rhf_energies.append(rhf.e_tot)
        uhf_energies.append(uhf.e_tot)
        fci_energies.append(fci_energy)
        qsci_energies.append(qsci_energy)

    # Plot energies vs bond length
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    plt.plot(bond_lengths, rhf_energies, 'o-', label='RHF', linewidth=2, markersize=8)
    plt.plot(bond_lengths, uhf_energies, 'o-', label='UHF', linewidth=2, markersize=8)
    plt.plot(bond_lengths, fci_energies, 's-', label='FCI', linewidth=2, markersize=8)
    plt.plot(bond_lengths, qsci_energies, '^-', label='QSCI', linewidth=2, markersize=8)

    plt.xlabel('Bond Length (Angstrom)', fontsize=12)
    plt.ylabel('Energy (Hartree)', fontsize=12)
    plt.title('H4 Chain: Energy vs Bond Length', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('figures/h4_chain_energies.png', dpi=300, bbox_inches='tight')
    # plt.show()

    print("\nPlot saved as 'h4_chain_energies.png'")



def build_h4_chain(bond_length):
    """
    Build a linear chain of 4 hydrogen atoms with equal spacing.

    Creates a PySCF molecule object for a linear H4 chain with atoms
    positioned along the x-axis at x = 0, bond_length, 2*bond_length,
    and 3*bond_length. The molecule is constructed with STO-3G basis set,
    neutral charge, and singlet spin state.

    Parameters
    ----------
    bond_length : float
        Distance between adjacent hydrogen atoms in Angstroms.

    Returns
    -------
    mol : gto.Mole
        PySCF molecule object representing the H4 chain with the
        following properties:
        - basis: 'sto-3g'
        - charge: 0
        - spin: 0 (singlet)
        - 4 hydrogen atoms in linear configuration
    """
    coords = [
        (0 * bond_length, 0, 0),
        (1 * bond_length, 0, 0),
        (2 * bond_length, 0, 0),
        (3 * bond_length, 0, 0),
    ]
    geometry = '; '.join([f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in coords])
    mol = gto.Mole()
    mol.build(
        atom=geometry,
        unit='Angstrom',
        basis='sto-3g',
        charge=0,
        spin=0,
        verbose=0,
    )
    return mol


def calc_fci_energy(rhf):
    """
    Calculate Full Configuration Interaction (FCI) energy.

    Uses PySCF's FCI solver to compute the exact ground state energy
    within the given basis set. FCI provides the numerically exact
    solution to the Schrödinger equation for the given orbital basis.

    Parameters
    ----------
    rhf : scf.RHF
        Converged RHF calculation object containing molecular orbitals
        and electron integrals.

    Returns
    -------
    float
        FCI ground state energy in Hartree.

    Notes
    -----
    FCI scales exponentially with system size and is only feasible
    for small molecules (typically < 20 orbitals).
    """
    ci_solver = fci.FCI(rhf)
    fci_energy, fci_vec = ci_solver.kernel()
    return fci_energy


def calc_qsci_energy(H, statevector):
    """
    Calculate Quantum Subspace Configuration Interaction (QSCI) energy.

    Extracts the significant configurations from the statevector (based on
    amplitude threshold), constructs the Hamiltonian in this reduced subspace,
    and solves for the ground state energy. This provides a variational energy
    estimate using the quantum-circuit-prepared subspace.

    Parameters
    ----------
    H : scipy.sparse matrix
        Full Hamiltonian matrix in the computational basis (Fock space).
    statevector : circuit.Statevector
        Quantum statevector with amplitudes for all basis configurations.
        The `data` attribute contains the complex amplitude array.

    Returns
    -------
    float
        QSCI ground state energy in Hartree.

    Notes
    -----
    - Configurations with |amplitude| < 1e-12 are filtered out
    - For small subspaces (≤2 dimensions), uses dense eigenvalue solver
    - For larger subspaces, uses sparse eigenvalue solver (eigsh)
    - The QSCI energy is variational: E_QSCI ≥ E_FCI
    """
    idx = np.argwhere(np.abs(statevector.data) > 1e-12).ravel()
    H_sub = H[np.ix_(idx, idx)]

    # Handle small matrices where eigsh would fail
    # This occurs when uhf = rhf so the statevector is a single determinant
    if H_sub.shape[0] <= 2:
        # Should equal HF energy but worth checking
        eigenvalues, eigenvectors = eigh(H_sub.toarray())
        E0 = eigenvalues[0]
    else:
        E0, psi0 = eigsh(H_sub, k=1, which='SA')
        E0 = E0[0]

    return E0


def analyse_statevector(mol, rhf, uhf, statevector):
    """
    Analyze and compare properties of the quantum statevector.

    Performs comprehensive analysis of the statevector including:
    1. Configuration decomposition (bitstring representation)
    2. Energy comparison (RHF, UHF, FCI, QSCI, statevector expectation)
    3. Spin expectation values (S² operator)
    4. Spin symmetry preservation checks
    5. Energy with spin-symmetric subspace

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object containing geometry and basis information.
    rhf : scf.RHF
        Converged RHF calculation object.
    uhf : scf.UHF
        Converged UHF calculation object (typically from symmetry breaking).
    statevector : circuit.Statevector
        Quantum statevector from circuit simulation, representing the
        quantum state in the computational basis.

    Prints
    ------
    - DataFrame of dominant configurations (bitstrings and amplitudes)
    - Energy values from different methods (FCI, RHF, UHF, statevector)
    - Spin expectation values (S²) from different methods
    - Spin symmetry preservation check
    - Number of configurations with spin symmetry
    - QSCI energy in spin-symmetric subspace

    Notes
    -----
    Bitstrings use the convention: '0' = occupied, '1' = unoccupied.
    The first half of bits represent spin-up orbitals, second half spin-down.
    """
    # Analyze statevector
    sv_abs = np.abs(statevector.data)
    idx_sorted = np.argsort(sv_abs)[::-1]
    sv_abs = sv_abs[idx_sorted]

    bitstring_ints = np.arange(len(sv_abs))[idx_sorted]
    bitstrings = [f"{x:0{2 * mol.nao}b}" for x in bitstring_ints]

    df = pd.DataFrame({'Bitstring': bitstrings, 'Psi abs': sv_abs})
    df = df.round(2)
    print(df)

    # Energy comparison
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)
    sv_energy = (statevector.data.conj() @ H @ statevector.data).real
    uhf_energy = uhf.e_tot
    rhf_energy = rhf.e_tot

    ci_solver = fci.FCI(rhf)
    fci_energy, fci_vec = ci_solver.kernel()
    fci_s2, mult = ci_solver.spin_square(fci_vec, mol.nao, mol.nelec)

    print("FCI energy:", fci_energy)
    print("RHF energy:", rhf_energy)
    print("UHF energy:", uhf_energy)
    print("Statevector energy (expectation):", sv_energy)

    # Spin comparison
    idx = np.argwhere(np.abs(statevector.data) > 1e-12).ravel()
    H_sub = H[np.ix_(idx, idx)]
    E0, psi0 = eigsh(H_sub, k=1, which='SA')

    S2_fermi = spin.total_spin_S2(mol.nao)
    S2_sub = S2_fermi[np.ix_(idx, idx)]
    qsci_s2 = (psi0.conj().T @ S2_sub @ psi0).real

    uhf_s2, uhf_multiplicity = uhf.spin_square()
    rhf_bitstring = int(f"{'0' * mol.nelectron}{'1' * (2 * mol.nao - mol.nelectron)}", 2)
    rhf_s2 = S2_fermi[rhf_bitstring, rhf_bitstring].real

    print("FCI spin:", fci_s2)
    print("RHF spin:", rhf_s2)
    print("UHF spin:", uhf_s2)
    print("QSCI spin:", qsci_s2[0, 0].real if qsci_s2.ndim > 1 else qsci_s2.real)

    # Check spin symmetry
    sampled_configs, symm_configs = utils.find_spin_symmetric_configs(n_bits=2 * mol.nao, idx=idx)
    print("Spin symmetry preserved:", np.array_equal(sampled_configs, symm_configs))

    idx_symm = [int(x, 2) for x in symm_configs]
    idx_symm = sorted(set(idx_symm))
    H_sub_symm = H[np.ix_(idx_symm, idx_symm)]
    E0_symm, psi0_symm = eigsh(H_sub_symm, k=1, which='SA')

    print("Configs with symmetry:", len(idx_symm))
    print("QSCI energy (symmetric):", E0_symm[0])


if __name__ == "__main__":
    main()
