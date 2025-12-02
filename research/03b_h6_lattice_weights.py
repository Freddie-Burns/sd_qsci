"""
H6 Triangular Lattice QSCI Convergence Analysis
================================================

This script performs a comprehensive analysis of Quantum Selected Configuration
Interaction (QSCI) convergence for a 6-hydrogen triangular lattice system.

Overview
--------
For a fixed bond length (2.0 A), the script:
1. Runs RHF, UHF, and FCI calculations using PySCF
2. Constructs the UHF statevector via orbital rotation circuit
3. Analyzes how QSCI energy converges to FCI by varying the number of
   configurations included based on amplitude ranking from the UHF statevector

The key insight is that configurations with large amplitudes in the UHF
statevector (obtained via orbital rotation) provide good approximations to
the FCI ground state energy, even though they're selected from UHF rather
than FCI amplitudes.

Methodology
-----------
QSCI Energy Calculation:
    - Select the n configurations with largest amplitudes from UHF statevector
    - Construct Hamiltonian subspace using these configurations
    - Diagonalize to get ground state energy in the subspace
    - Compare to FCI energy to assess convergence

FCI Subspace Energy (Baseline):
    - Same procedure but using FCI amplitudes for configuration selection
    - Provides an upper bound on achievable convergence rate

Mean Sample Number:
    - For each subspace size, calculate 1/(min_coeff)^2
    - Represents expected number of samples needed to observe the least
      probable configuration in the subspace
    - Provides a quantum sampling complexity metric

Outputs
-------
CSV Files:
    - h6_qsci_convergence.csv: Full convergence data with columns:
        * subspace_size: Number of configurations in subspace
        * qsci_energy: Energy using UHF-based selection
        * fci_subspace_energy: Energy using FCI-based selection
        * mean_sample_number: Sampling complexity metric
    - h6_summary.csv: Summary statistics including:
        * Reference energies (RHF, UHF, FCI)
        * Milestone achievements (configs to reach UHF/FCI)
        * Maximum subspace size
        * Energy differences

Plots:
    - h6_energy_vs_samples.png: Energy vs mean sample number (log scale)
    - h6_qsci_convergence.png: Energy vs subspace size for QSCI and FCI
    - statevector_coefficients.png: Top 20 coefficient comparison (bar chart)
    - statevector_coefficients_full.png: All significant coefficients (log scale)

Key Results
-----------
The analysis tracks two important milestones:
1. Number of configurations needed to fall below UHF energy
2. Number of configurations needed to reach FCI energy (within tolerance)

These metrics help assess the efficiency of UHF-based configuration selection
for approximating FCI results in quantum computing applications.

Notes
-----
- Uses STO-3G basis set
- Singlet (spin=0) ground state
- BLOCK spin ordering: [α0...α(nmo-1), β0...β(nmo-1)]
- Configuration selection based on absolute amplitude values
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, scf, fci
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from sd_qsci.utils import uhf_from_rhf
from sd_qsci import circuit, hamiltonian


SV_TOL = 1e-6  # Tolerance for including a configuration amplitude
FCI_TOL = 1e-6  # Tolerance for considering energy "reached FCI"


@dataclass
class QuantumChemistryResults:
    """Container for quantum chemistry calculation results."""
    mol: gto.Mole
    rhf: scf.RHF
    uhf: scf.UHF
    sv: circuit.Statevector
    H: np.ndarray  # Hamiltonian matrix
    fci_energy: float
    n_fci_configs: int
    fci_vec: np.ndarray
    bond_length: float


@dataclass
class ConvergenceResults:
    """Container for convergence analysis results."""
    df: pd.DataFrame  # Contains: subspace_size, qsci_energy, fci_subspace_energy, mean_sample_number
    max_size: int
    n_configs_below_uhf: Optional[int]
    n_configs_reach_fci: Optional[int]


def main():
    """
    Run H6 triangular lattice energy calculations and analyze QSCI convergence.

    For a fixed bond length, performs RHF, UHF, and FCI calculations, then
    analyzes how QSCI energy converges to FCI as the subspace size increases
    by varying the number of configurations included based on amplitude ranking.
    """
    # Setup
    data_dir = setup_data_directory()
    bond_length = 2.0
    print(f"Running bond length: {bond_length:.2f} Angstrom")

    # Run quantum chemistry calculations
    mol = build_h6_lattice(bond_length)
    rhf = scf.RHF(mol).run()
    qc_results = run_quantum_chemistry_calculations(mol, rhf, bond_length)

    # Calculate convergence data
    conv_results = calculate_convergence_data(qc_results)

    # Save data to CSV
    save_convergence_data(data_dir, qc_results, conv_results)

    # Create plots
    plot_energy_vs_samples(data_dir, qc_results, conv_results)
    plot_convergence_comparison(data_dir, qc_results, conv_results)

    # Compute final QSCI wavefunction and plot coefficients
    print(f"\nComputing QSCI ground state wavefunction with {conv_results.max_size} configurations...")
    qsci_energy_final, qsci_vec, qsci_indices = calc_qsci_energy_with_size(
        qc_results.H, qc_results.sv, conv_results.max_size, return_vector=True)
    plot_statevector_coefficients(qc_results.sv.data, qc_results.fci_vec, data_dir, n_top=20)

    # Print summary
    print_summary(data_dir, qc_results, conv_results, qsci_energy_final)

def setup_data_directory():
    """
    Create and return the data directory for this script.

    Returns
    -------
    data_dir : Path
        Directory path for saving output files.
    """
    script_name = Path(__file__).stem
    data_dir = Path(__file__).parent / 'data' / script_name
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def run_quantum_chemistry_calculations(mol, rhf, bond_length):
    """
    Run UHF and FCI calculations and build Hamiltonian.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object.
    rhf : scf.RHF
        Converged RHF calculation object.
    bond_length : float
        Bond length in Angstroms.

    Returns
    -------
    qc_results : QuantumChemistryResults
        Container with all quantum chemistry calculation results.
    """
    uhf = uhf_from_rhf(mol, rhf)
    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
    sv = circuit.simulate(qc)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)

    # Verify orbital rotation worked
    sv_energy = (sv.data.conj().T @ H @ sv.data).real
    assert np.isclose(sv_energy, uhf.e_tot), "Orbital rotation verification failed"

    fci_energy, n_fci_configs, fci_vec = calc_fci_energy(rhf)

    return QuantumChemistryResults(
        mol=mol,
        rhf=rhf,
        uhf=uhf,
        sv=sv,
        H=H,
        fci_energy=fci_energy,
        n_fci_configs=n_fci_configs,
        fci_vec=fci_vec,
        bond_length=bond_length
    )


def calculate_convergence_data(qc_results: QuantumChemistryResults):
    """
    Calculate QSCI and FCI subspace energies for varying subspace sizes.

    Parameters
    ----------
    qc_results : QuantumChemistryResults
        Container with quantum chemistry calculation results.

    Returns
    -------
    conv_results : ConvergenceResults
        Container with convergence analysis results including DataFrame.
    """
    # Determine maximum subspace size
    max_idx = np.argwhere(np.abs(qc_results.sv.data) > SV_TOL).ravel()
    max_size = len(max_idx)
    print(f"Maximum subspace size (amplitude > {SV_TOL}): {max_size}")

    subspace_sizes = list(range(1, max_size + 1))
    qsci_energies = []
    fci_subspace_energies = []
    mean_sample_numbers = []

    # Track milestones
    n_configs_below_uhf = None
    n_configs_reach_fci = None

    print(f"Calculating energies for {len(subspace_sizes)} subspace sizes...")
    for size in subspace_sizes:
        # QSCI energy (based on UHF statevector amplitudes)
        energy = calc_qsci_energy_with_size(qc_results.H, qc_results.sv, size)
        qsci_energies.append(energy)

        # FCI subspace energy (based on FCI amplitudes)
        fci_sub_energy = calc_fci_subspace_energy(qc_results.H, qc_results.fci_vec, size)
        fci_subspace_energies.append(fci_sub_energy)

        # Calculate mean sample number
        idx = np.argsort(np.abs(qc_results.sv.data))[-size:]
        min_coeff = np.min(np.abs(qc_results.sv.data[idx]))
        mean_sample_number = 1.0 / (min_coeff ** 2)
        mean_sample_numbers.append(mean_sample_number)

        # Check milestones
        if n_configs_below_uhf is None and energy < qc_results.uhf.e_tot:
            n_configs_below_uhf = size

        if n_configs_reach_fci is None and abs(energy - qc_results.fci_energy) < FCI_TOL:
            n_configs_reach_fci = size

    # Create DataFrame with all convergence data
    df = pd.DataFrame({
        'subspace_size': subspace_sizes,
        'qsci_energy': qsci_energies,
        'fci_subspace_energy': fci_subspace_energies,
        'mean_sample_number': mean_sample_numbers
    })

    return ConvergenceResults(
        df=df,
        max_size=max_size,
        n_configs_below_uhf=n_configs_below_uhf,
        n_configs_reach_fci=n_configs_reach_fci
    )


def save_convergence_data(data_dir: Path, qc_results: QuantumChemistryResults,
                          conv_results: ConvergenceResults):
    """
    Save convergence data and summary statistics to CSV files.

    Parameters
    ----------
    data_dir : Path
        Directory to save the CSV files.
    qc_results : QuantumChemistryResults
        Container with quantum chemistry calculation results.
    conv_results : ConvergenceResults
        Container with convergence analysis results.
    """
    # Save convergence data
    conv_results.df.to_csv(data_dir / 'h6_qsci_convergence.csv', index=False)

    # Save summary statistics
    summary_data = {
        'bond_length': qc_results.bond_length,
        'rhf_energy': qc_results.rhf.e_tot,
        'uhf_energy': qc_results.uhf.e_tot,
        'fci_energy': qc_results.fci_energy,
        'n_fci_configs': qc_results.n_fci_configs,
        'n_configs_below_uhf': conv_results.n_configs_below_uhf if conv_results.n_configs_below_uhf else 'Never',
        'n_configs_reach_fci': conv_results.n_configs_reach_fci if conv_results.n_configs_reach_fci else 'Never',
        'max_subspace_size': conv_results.max_size,
        'min_qsci_energy': conv_results.df['qsci_energy'].min(),
        'energy_diff_to_fci': conv_results.df['qsci_energy'].min() - qc_results.fci_energy
    }
    summary_df = pd.DataFrame(list(summary_data.items()), columns=['quantity', 'value'])
    summary_df.to_csv(data_dir / 'h6_summary.csv', index=False)


def plot_energy_vs_samples(data_dir: Path, qc_results: QuantumChemistryResults,
                           conv_results: ConvergenceResults):
    """
    Create and save energy vs mean sample number plot.

    Parameters
    ----------
    data_dir : Path
        Directory to save the plot.
    qc_results : QuantumChemistryResults
        Container with quantum chemistry calculation results.
    conv_results : ConvergenceResults
        Container with convergence analysis results.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.semilogx(conv_results.df['mean_sample_number'], conv_results.df['qsci_energy'], 'o-',
                label='QSCI (UHF-based selection)',
                linewidth=2, markersize=4, color='purple')

    # Add horizontal reference lines
    ax.axhline(y=qc_results.rhf.e_tot, color='blue', linestyle='--',
               linewidth=2, label=f'RHF: {qc_results.rhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.uhf.e_tot, color='orange', linestyle='--',
               linewidth=2, label=f'UHF: {qc_results.uhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.fci_energy, color='green', linestyle='--',
               linewidth=2, label=f'FCI: {qc_results.fci_energy:.6f} Ha')

    ax.set_xlabel('Mean Sample Number (log scale)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    ax.set_title(
        f'H$_6$ Triangular Lattice Simulation: Energy vs Mean Sample Number\nBond Length = {qc_results.bond_length:.2f} Å',
        fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(data_dir / 'h6_energy_vs_samples.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_convergence_comparison(data_dir: Path, qc_results: QuantumChemistryResults,
                                conv_results: ConvergenceResults):
    """
    Create and save convergence comparison plot.

    Parameters
    ----------
    data_dir : Path
        Directory to save the plot.
    qc_results : QuantumChemistryResults
        Container with quantum chemistry calculation results.
    conv_results : ConvergenceResults
        Container with convergence analysis results.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot QSCI energy vs subspace size
    ax.plot(conv_results.df['subspace_size'], conv_results.df['qsci_energy'], 'o-',
            label='QSCI (UHF-based selection)',
            linewidth=2, markersize=4, color='purple')

    # Plot FCI subspace energy vs subspace size
    ax.plot(conv_results.df['subspace_size'], conv_results.df['fci_subspace_energy'], 's-',
            label='FCI subspace (FCI-based selection)',
            linewidth=2, markersize=4, color='darkgreen')

    # Add horizontal reference lines
    ax.axhline(y=qc_results.rhf.e_tot, color='blue', linestyle='--',
               linewidth=2, label=f'RHF: {qc_results.rhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.uhf.e_tot, color='orange', linestyle='--',
               linewidth=2, label=f'UHF: {qc_results.uhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.fci_energy, color='green', linestyle='--',
               linewidth=2, label=f'FCI: {qc_results.fci_energy:.6f} Ha')

    ax.set_xlabel('Subspace Size (Number of Configurations)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    ax.set_title(f'H6 Triangular Lattice: Energy Convergence Comparison\nBond Length = {qc_results.bond_length:.2f} Å',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(data_dir / 'h6_qsci_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(data_dir: Path, qc_results: QuantumChemistryResults,
                 conv_results: ConvergenceResults, qsci_energy_final: float):
    """
    Print summary of results to console.

    Parameters
    ----------
    data_dir : Path
        Directory where data was saved.
    qc_results : QuantumChemistryResults
        Container with quantum chemistry calculation results.
    conv_results : ConvergenceResults
        Container with convergence analysis results.
    qsci_energy_final : float
        Final QSCI energy with max subspace.
    """
    print(f"\nReference Energies:")
    print(f"  RHF: {qc_results.rhf.e_tot:.8f} Ha")
    print(f"  UHF: {qc_results.uhf.e_tot:.8f} Ha")
    print(f"  FCI: {qc_results.fci_energy:.8f} Ha")
    print(f"  QSCI (max subspace): {qsci_energy_final:.8f} Ha")
    print(f"\nFCI Solution:")
    print(f"  Number of configurations: {qc_results.n_fci_configs}")
    print(f"\nQSCI Convergence:")
    print(f"  Max subspace size: {conv_results.max_size}")
    print(f"  Min QSCI energy: {conv_results.df['qsci_energy'].min():.8f} Ha")
    print(f"  Energy difference to FCI: {conv_results.df['qsci_energy'].min() - qc_results.fci_energy:.2e} Ha")
    print(f"\nMilestones:")
    print(f"  Configs to fall below UHF: {conv_results.n_configs_below_uhf if conv_results.n_configs_below_uhf else 'Never achieved'}")
    print(f"  Configs to reach FCI (±{FCI_TOL:.0e} Ha): {conv_results.n_configs_reach_fci if conv_results.n_configs_reach_fci else 'Never achieved'}")
    print(f"\nData saved to '{data_dir}' directory:")
    print("  - h6_qsci_convergence.csv (full energy data)")
    print("  - h6_summary.csv (summary statistics)")
    print("  - h6_qsci_convergence.png (plot)")
    print("  - h6_energy_vs_samples.png (energy vs mean sample number)")
    print("  - statevector_coefficients.png (top 20 coefficients bar chart)")
    print("  - statevector_coefficients_full.png (all significant coefficients)")


def calc_qsci_energy_with_size(H, statevector, n_configs, return_vector=False):
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
    return_vector : bool, optional
        If True, return the eigenvector in full space along with energy.

    Returns
    -------
    E0 : float
        QSCI ground state energy in Hartree.
    psi0_full : np.ndarray, optional
        QSCI ground state wavefunction in the full Fock space (only if return_vector=True).
    idx : np.ndarray, optional
        Indices of configurations included in QSCI subspace (only if return_vector=True).
    """
    # Get indices of n_configs largest amplitude configurations
    idx = np.argsort(np.abs(statevector.data))[-n_configs:]
    H_sub = H[np.ix_(idx, idx)]

    # Handle small matrices where eigsh would fail
    if H_sub.shape[0] <= 2:
        eigenvalues, eigenvectors = eigh(H_sub.toarray())
        E0 = eigenvalues[0]
        psi0 = eigenvectors[:, 0]
    else:
        E0, psi0 = eigsh(H_sub, k=1, which='SA')
        E0 = E0[0]
        psi0 = psi0[:, 0]

    if return_vector:
        # Build full space vector
        psi0_full = np.zeros(statevector.data.shape, dtype=complex)
        psi0_full[idx] = psi0
        return E0, psi0_full, idx

    return E0


def calc_fci_subspace_energy(H, fci_vec, n_configs):
    """
    Calculate energy in subspace of n largest amplitude FCI configurations.

    Parameters
    ----------
    H : scipy.sparse matrix
        Full Hamiltonian matrix in the computational basis (Fock space).
    fci_vec : np.ndarray
        Full FCI wavefunction vector (flattened).
    n_configs : int
        Number of configurations to include in the subspace.

    Returns
    -------
    E0 : float
        Ground state energy in the FCI subspace in Hartree.
    """
    # Get indices of n_configs largest amplitude FCI configurations
    idx = np.argsort(np.abs(fci_vec))[-n_configs:]
    H_sub = H[np.ix_(idx, idx)]

    # Handle small matrices where eigsh would fail
    if H_sub.shape[0] <= 2:
        eigenvalues = eigh(H_sub.toarray(), eigvals_only=True)
        E0 = eigenvalues[0]
    else:
        E0 = eigsh(H_sub, k=1, which='SA', return_eigenvectors=False)
        E0 = E0[0]

    return E0


def plot_statevector_coefficients(qsci_vec, fci_vec, data_dir, n_top=20):
    """
    Plot bar charts comparing QSCI and FCI statevector coefficients.

    Parameters
    ----------
    qsci_vec : np.ndarray
        QSCI ground state wavefunction in full Fock space.
    fci_vec : np.ndarray
        FCI ground state wavefunction in full Fock space.
    data_dir : Path
        Directory to save the plot.
    n_top : int, optional
        Number of top configurations to display (by FCI amplitude).
    """
    # Get indices of n_top largest FCI coefficients
    fci_abs = np.abs(fci_vec)
    top_indices = np.argsort(fci_abs)[-n_top:][::-1]  # Descending order

    # Extract coefficients for these indices
    qsci_coefs = np.abs(qsci_vec[top_indices])
    fci_coefs = fci_abs[top_indices]

    # Create bar chart
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(n_top)
    width = 0.35

    # Plot bars
    ax.bar(x - width/2, fci_coefs, width, label='FCI', color='green', alpha=0.8)
    ax.bar(x + width/2, qsci_coefs, width, label='QSCI', color='purple', alpha=0.8)

    ax.set_xlabel('Configuration Index (sorted by FCI amplitude)', fontsize=12)
    ax.set_ylabel('|Coefficient|', fontsize=12)
    ax.set_title(f'H$_6$ Lattice Comparison of FCI and QSCI Statevector Coefficients\nTop {n_top} Configurations',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in top_indices], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(data_dir / 'statevector_coefficients.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Also create a plot showing all significant coefficients (log scale)
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    # Get all non-zero indices from both vectors
    significant_mask = (fci_abs > 1e-10) | (np.abs(qsci_vec) > 1e-10)
    significant_indices = np.where(significant_mask)[0]

    # Sort by FCI amplitude
    sort_order = np.argsort(fci_abs[significant_indices])[::-1]
    sorted_indices = significant_indices[sort_order]

    fci_sig = fci_abs[sorted_indices]
    qsci_sig = np.abs(qsci_vec[sorted_indices])

    x_all = np.arange(len(sorted_indices))

    ax2.semilogy(x_all, fci_sig, 'o-', label='FCI', color='green', markersize=3, linewidth=1)
    ax2.semilogy(x_all, qsci_sig, 's-', label='QSCI', color='purple', markersize=3, linewidth=1, alpha=0.7)

    ax2.set_xlabel('Configuration Index (sorted by FCI amplitude)', fontsize=12)
    ax2.set_ylabel('|Coefficient| (log scale)', fontsize=12)
    ax2.set_title('FCI vs QSCI Statevector Coefficients (All Significant Configurations)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(data_dir / 'statevector_coefficients_full.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print(f"\nStatevector Coefficient Analysis:")
    print(f"  Number of significant FCI coefficients: {np.sum(fci_abs > 1e-10)}")
    print(f"  Number of significant QSCI coefficients: {np.sum(np.abs(qsci_vec) > 1e-10)}")
    print(f"  Max FCI coefficient: {np.max(fci_abs):.6f}")
    print(f"  Max QSCI coefficient: {np.max(np.abs(qsci_vec)):.6f}")
    print(f"  Overlap <FCI|QSCI>: {np.abs(np.vdot(fci_vec, qsci_vec)):.6f}")


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


def fci_to_fock_space(fci_vec, mol, nelec):
    """
    Convert PySCF FCI vector (CI space) to full Fock space vector.

    PySCF's FCI vector only contains configurations with the correct number
    of electrons. This function maps it to the full 2^n Fock space where n
    is the number of spin orbitals.

    Parameters
    ----------
    fci_vec : np.ndarray
        FCI wavefunction from PySCF (shape: n_alpha_configs x n_beta_configs).
    mol : gto.Mole
        PySCF molecule object.
    nelec : tuple
        (n_alpha, n_beta) electron counts.

    Returns
    -------
    fock_vec : np.ndarray
        FCI wavefunction in full Fock space basis (length 2^(2*nmo)).
    """
    from pyscf.fci import cistring

    nmo = mol.nao  # number of spatial orbitals
    n_alpha, n_beta = nelec
    n_spin_orbitals = 2 * nmo

    # Generate all alpha and beta string indices
    alpha_strs = cistring.make_strings(range(nmo), n_alpha)
    beta_strs = cistring.make_strings(range(nmo), n_beta)

    # Initialize full Fock space vector (all 2^(2*nmo) configurations)
    fock_vec = np.zeros(2**n_spin_orbitals, dtype=complex)

    # Map FCI configurations to Fock space
    # In BLOCK spin ordering: [α0..α(nmo-1), β0..β(nmo-1)]
    fci_vec_flat = fci_vec.flatten()

    for i_alpha, alpha_str in enumerate(alpha_strs):
        for i_beta, beta_str in enumerate(beta_strs):
            # Convert string representations to Fock space index
            # alpha_str and beta_str are integers representing occupation patterns
            fock_idx = (alpha_str << nmo) | beta_str
            ci_idx = i_alpha * len(beta_strs) + i_beta
            fock_vec[fock_idx] = fci_vec_flat[ci_idx]

    return fock_vec


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
    fci_vec : np.ndarray
        FCI wavefunction vector in full Fock space (length 2^(2*nmo)).
    """
    ci_solver = fci.FCI(rhf)
    fci_energy, fci_vec_ci = ci_solver.kernel()

    # Get electron counts
    mol = rhf.mol
    nelec = mol.nelec

    # Convert from CI space to full Fock space
    fci_vec = fci_to_fock_space(fci_vec_ci, mol, nelec)

    # Count non-zero configurations in FCI wavefunction
    n_configs = np.count_nonzero(np.abs(fci_vec) > 1e-10)

    return fci_energy, n_configs, fci_vec


if __name__ == "__main__":
    main()
