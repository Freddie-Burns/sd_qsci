"""
Create H6 triangular lattice and compare energies from RHF, UHF, FCI, and QSCI.
Analyze QSCI energy convergence as a function of subspace size.
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, scf, fci
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import os
import pandas as pd
from pathlib import Path

from sd_qsci.utils import uhf_from_rhf
from sd_qsci import circuit, hamiltonian

SV_TOL = 1e-6
FCI_TOL = 1e-6  # Tolerance for considering energy "reached FCI"

def main():
    """
    Run H6 triangular lattice energy calculations and analyze QSCI convergence.

    For a fixed bond length, performs RHF, UHF, and FCI calculations, then
    analyzes how QSCI energy converges to FCI as the subspace size increases
    by varying the number of configurations included based on amplitude ranking.
    """
    # Create subdirectory within data/ named after this script's prefix
    script_name = Path(__file__).stem  # e.g., '03_h6_lattice_n_dets'
    data_dir = Path(__file__).parent / 'data' / script_name
    data_dir.mkdir(parents=True, exist_ok=True)

    bond_length = 2.0
    print(f"Running bond length: {bond_length:.2f} Angstrom")

    # Setup and run calculations
    mol = build_h6_lattice(bond_length)
    rhf = scf.RHF(mol).run()
    uhf = uhf_from_rhf(mol, rhf)
    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
    sv = circuit.simulate(qc)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)
    sv_energy = (sv.data.conj().T @ H @ sv.data).real
    assert np.isclose(sv_energy, uhf.e_tot)

    fci_energy, n_fci_configs = calc_fci_energy(rhf)

    # Determine maximum subspace size (configurations with amplitude > 0.01)
    max_idx = np.argwhere(np.abs(sv.data) > SV_TOL).ravel()
    max_size = len(max_idx)
    print(f"Maximum subspace size (amplitude > {SV_TOL}): {max_size}")

    # Calculate QSCI energies for different subspace sizes
    subspace_sizes = list(range(1, max_size + 1))
    qsci_energies = []

    # Track milestones
    n_configs_below_uhf = None
    n_configs_reach_fci = None

    for size in subspace_sizes:
        energy = calc_qsci_energy_with_size(H, sv, size)
        qsci_energies.append(energy)

        # Check if this is the first time we fall below UHF energy
        if n_configs_below_uhf is None and energy < uhf.e_tot:
            n_configs_below_uhf = size

        # Check if we've reached FCI energy (within tolerance)
        if n_configs_reach_fci is None and abs(energy - fci_energy) < FCI_TOL:
            n_configs_reach_fci = size

    # Save data to CSV
    data_df = pd.DataFrame({
        'subspace_size': subspace_sizes,
        'qsci_energy': qsci_energies
    })
    data_df.to_csv(data_dir / 'h6_qsci_convergence.csv', index=False)

    # Save summary statistics (transposed so each quantity is a row)
    summary_data = {
        'bond_length': bond_length,
        'rhf_energy': rhf.e_tot,
        'uhf_energy': uhf.e_tot,
        'fci_energy': fci_energy,
        'n_fci_configs': n_fci_configs,
        'n_configs_below_uhf': n_configs_below_uhf if n_configs_below_uhf else 'Never',
        'n_configs_reach_fci': n_configs_reach_fci if n_configs_reach_fci else 'Never',
        'max_subspace_size': max_size,
        'min_qsci_energy': min(qsci_energies),
        'energy_diff_to_fci': min(qsci_energies) - fci_energy
    }
    # Transpose: each quantity on a new row
    summary_df = pd.DataFrame(list(summary_data.items()), columns=['quantity', 'value'])
    summary_df.to_csv(data_dir / 'h6_summary.csv', index=False)

    # Create plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot QSCI energy vs subspace size
    ax.plot(subspace_sizes, qsci_energies, 'o-', label='QSCI',
            linewidth=2, markersize=4, color='purple')

    # Add horizontal reference lines
    ax.axhline(y=rhf.e_tot, color='blue', linestyle='--',
               linewidth=2, label=f'RHF: {rhf.e_tot:.6f} Ha')
    ax.axhline(y=uhf.e_tot, color='orange', linestyle='--',
               linewidth=2, label=f'UHF: {uhf.e_tot:.6f} Ha')
    ax.axhline(y=fci_energy, color='green', linestyle='--',
               linewidth=2, label=f'FCI: {fci_energy:.6f} Ha')

    ax.set_xlabel('Subspace Size (Number of Configurations)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    ax.set_title(f'H6 Triangular Lattice: QSCI Energy Convergence\nBond Length = {bond_length:.2f} Å',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(data_dir / 'h6_qsci_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nReference Energies:")
    print(f"  RHF: {rhf.e_tot:.8f} Ha")
    print(f"  UHF: {uhf.e_tot:.8f} Ha")
    print(f"  FCI: {fci_energy:.8f} Ha")
    print(f"\nFCI Solution:")
    print(f"  Number of configurations: {n_fci_configs}")
    print(f"\nQSCI Convergence:")
    print(f"  Max subspace size: {max(subspace_sizes)}")
    print(f"  Min QSCI energy: {min(qsci_energies):.8f} Ha")
    print(f"  Energy difference to FCI: {min(qsci_energies) - fci_energy:.2e} Ha")
    print(f"\nMilestones:")
    print(f"  Configs to fall below UHF: {n_configs_below_uhf if n_configs_below_uhf else 'Never achieved'}")
    print(f"  Configs to reach FCI (±{FCI_TOL:.0e} Ha): {n_configs_reach_fci if n_configs_reach_fci else 'Never achieved'}")
    print(f"\nData saved to '{data_dir}' directory:")
    print("  - h6_qsci_convergence.csv (full energy data)")
    print("  - h6_summary.csv (summary statistics)")
    print("  - h6_qsci_convergence.png (plot)")


def calc_qsci_energy_with_size(H, statevector, n_configs):
    """
    Calculate QSCI energy using the n largest amplitude configurations.

    Parameters
    ----------
    H : scipy.sparse matrix
        Full Hamiltonian matrix in the computational basis (Fock space).
    statevector : circuit.Statevector
        Quantum statevector with amplitudes for all basis configurations.
    n_configs : int
        Number of configurations to include in the QSCI subspace.

    Returns
    -------
    E0 : float
        QSCI ground state energy in Hartree.
    """
    # Get indices of n_configs largest amplitude configurations
    idx = np.argsort(np.abs(statevector.data))[-n_configs:]
    H_sub = H[np.ix_(idx, idx)]

    # Handle small matrices where eigsh would fail
    if H_sub.shape[0] <= 2:
        eigenvalues, eigenvectors = eigh(H_sub.toarray())
        E0 = eigenvalues[0]
    else:
        E0, psi0 = eigsh(H_sub, k=1, which='SA')
        E0 = E0[0]

    return E0


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
    fci_energy : float
        FCI ground state energy in Hartree.
    n_configs : int
        Number of configurations in the FCI wavefunction.
    """
    ci_solver = fci.FCI(rhf)
    fci_energy, fci_vec = ci_solver.kernel()

    # Count non-zero configurations in FCI wavefunction
    n_configs = np.count_nonzero(np.abs(fci_vec) > 1e-10)

    return fci_energy, n_configs


if __name__ == "__main__":
    main()
