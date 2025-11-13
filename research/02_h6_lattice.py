"""
Create H6 triangular lattice and compare energies from RHF, UHF, FCI, and QSCI.
Ensure the statevector after orbital rotation gives the same energy and spin
as the input UHF state.
"""

import numpy as np
import seaborn as sns
from line_profiler import profile
from matplotlib import pyplot as plt
from pyscf import gto, scf, fci
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from sd_qsci.utils import uhf_from_rhf
from sd_qsci import circuit, hamiltonian


@profile
def main():
    """
    Run H6 triangular lattice energy calculations across multiple bond lengths.

    Performs RHF, UHF, FCI, and QSCI calculations on a H6 triangular lattice
    for bond lengths ranging from 0.5 to 3.0 Angstrom. Generates two plots:
    1. Energy comparison across different methods
    2. QSCI subspace dimension vs bond length

    Notes
    -----
    Asserts that the statevector energy matches the UHF energy to validate
    the orbital rotation circuit implementation.
    """
    rhf_energies = []
    uhf_energies = []
    fci_energies = []
    qsci_energies = []
    qsci_subspace_dims = []
    bond_lengths = []

    for bond_length in np.linspace(2, 2.5, 2):
        print(f"Running bond length: {bond_length:.2f} Angstrom")
        mol = build_h6_lattice(bond_length)
        rhf = scf.RHF(mol).run()
        uhf = uhf_from_rhf(mol, rhf)
        qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
        sv = circuit.simulate(qc)
        H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)
        sv_energy = (sv.data.conj().T @ H @ sv.data).real
        assert np.isclose(sv_energy, uhf.e_tot)

        fci_energy = calc_fci_energy(rhf)
        qsci_energy, qsci_idx = calc_qsci_energy(H, sv)

        bond_lengths.append(bond_length)
        rhf_energies.append(rhf.e_tot)
        uhf_energies.append(uhf.e_tot)
        fci_energies.append(fci_energy)
        qsci_energies.append(qsci_energy)
        qsci_subspace_dims.append(len(qsci_idx))

    # Plot energies vs bond length
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Energy comparison
    ax1.plot(bond_lengths, rhf_energies, 'o-', label='RHF', linewidth=2, markersize=8)
    ax1.plot(bond_lengths, uhf_energies, 'o-', label='UHF', linewidth=2, markersize=8)
    ax1.plot(bond_lengths, fci_energies, 's-', label='FCI', linewidth=2, markersize=8)
    ax1.plot(bond_lengths, qsci_energies, '^-', label='QSCI', linewidth=2, markersize=8)
    ax1.set_xlabel('Bond Length (Angstrom)', fontsize=12)
    ax1.set_ylabel('Energy (Hartree)', fontsize=12)
    ax1.set_title('H6 Triangular Lattice: Energy vs Bond Length', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: QSCI subspace dimension
    ax2.plot(bond_lengths, qsci_subspace_dims, 'o-', linewidth=2, markersize=8, color='purple')
    ax2.set_xlabel('Bond Length (Angstrom)', fontsize=12)
    ax2.set_ylabel('Number of Configurations', fontsize=12)
    ax2.set_title('QSCI Subspace Dimension vs Bond Length', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig('figures/h6_lattice_energies.png', dpi=300, bbox_inches='tight')
    # plt.show()

    print("\nPlot saved as 'figures/h6_lattice_energies.png'")



def build_h6_lattice(bond_length):
    """
    Build a triangular lattice of 6 hydrogen atoms.

    Creates a PySCF molecule object for 6 hydrogen atoms arranged in a
    triangular lattice pattern. The atoms form two rows: the first row
    has 3 atoms and the second row has 3 atoms, arranged such that each
    atom in the second row sits between two atoms in the first row,
    creating a triangular lattice structure.

    Parameters
    ----------
    bond_length : float
        Distance between adjacent hydrogen atoms in Angstroms.

    Returns
    -------
    mol : gto.Mole
        PySCF molecule object representing the H6 triangular lattice with
        the following properties:
        - basis: 'sto-3g'
        - charge: 0
        - spin: 0 (singlet)
        - 6 hydrogen atoms in triangular lattice configuration
    """
    # Create triangular lattice: 3 atoms in first row, 3 in second row
    # First row: atoms at x = 0, 1*bond_length, 2*bond_length
    # Second row: atoms offset by bond_length/2 in x and sqrt(3)/2*bond_length in y
    h = bond_length * np.sqrt(3) / 2
    coords = [
        # First row
        (0.0 * bond_length, 0, 0),
        (1.0 * bond_length, 0, 0),
        (2.0 * bond_length, 0, 0),
        (0.5 * bond_length, h, 0),
        (1.5 * bond_length, h, 0),
        (2.0 * bond_length, 2 * h, 0),
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


def calc_qsci_energy(H, statevector):
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


if __name__ == "__main__":
    main()
