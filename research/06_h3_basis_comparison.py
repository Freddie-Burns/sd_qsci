"""
H3+ Triangular Hydrogen Ion - Basis Set Comparison.

Compare QSCI convergence across multiple basis sets:
- STO-3G
- STO-6G
- 3-21G
- 6-31G
- 6-31G*
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, scf

from sd_qsci import analysis


# Configuration
BASIS_SETS = [
    "sto-3g",
    "sto-6g",
    "3-21g",
    "6-31g",
    "6-31g*",
    "6-31g**",
    "6-311++g**",
    "cc-pvdz",
    "cc-pvtz",
    # "aug-cc-pvtz",
    # "aug-cc-pvqz"
]
BOND_LENGTH = 2.0
SV_TOL = 1e-12
FCI_TOL = 1e-6


def main():
    """
    Run H3+ calculations for multiple basis sets and compare convergence.
    """
    script_name = Path(__file__).stem
    data_dir = Path(__file__).parent / 'data' / script_name
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"H3+ Basis Set Comparison Study")
    print(f"=" * 60)
    print(f"Bond length: {BOND_LENGTH:.2f} Angstrom")
    print(f"Basis sets: {', '.join(BASIS_SETS)}")
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    # Store results for each basis set
    all_results = {}

    # Run calculations for each basis set
    for basis in BASIS_SETS:
        print(f"\n{'=' * 60}")
        print(f"Processing basis set: {basis}")
        print(f"{'=' * 60}")

        try:
            results = run_single_basis_calculation(basis)
            all_results[basis] = results
            print(f"✓ Successfully completed {basis}")
        except Exception as e:
            print(f"✗ Error with {basis}: {e}")
            continue

    if not all_results:
        print("\nNo successful calculations. Exiting.")
        return

    # Save individual basis set data
    print(f"\n{'=' * 60}")
    print("Saving individual basis set data...")
    print(f"{'=' * 60}")
    save_individual_data(data_dir, all_results)

    # Create comparison plots
    print(f"\n{'=' * 60}")
    print("Creating comparison plots...")
    print(f"{'=' * 60}")
    plot_basis_comparison(data_dir, all_results)
    plot_energy_levels_comparison(data_dir, all_results)

    # Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print_summary_table(all_results)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete! Results saved to:")
    print(f"  {data_dir}")
    print(f"{'=' * 60}")


def run_single_basis_calculation(basis: str) -> Dict:
    """
    Run quantum chemistry calculations for a single basis set.

    Parameters
    ----------
    basis : str
        Basis set name.

    Returns
    -------
    dict
        Dictionary containing qc_results and conv_results.
    """
    # Build molecule
    mol = build_h3_plus(BOND_LENGTH, basis)

    # Run RHF
    print(f"  Running RHF...")
    rhf = scf.RHF(mol).run(verbose=0)

    # Run full quantum chemistry workflow
    print(f"  Running quantum chemistry calculations...")
    qc_results = analysis.run_quantum_chemistry_calculations(mol, rhf, BOND_LENGTH)

    # Calculate convergence data
    print(f"  Calculating convergence data...")
    conv_results = analysis.calc_convergence_data(
        qc_results,
        sv_tol=SV_TOL,
        fci_tol=FCI_TOL
    )

    # Print quick stats
    print(f"  RHF Energy: {qc_results.rhf.e_tot:.8f} Ha")
    print(f"  UHF Energy: {qc_results.uhf.e_tot:.8f} Ha")
    print(f"  FCI Energy: {qc_results.fci_energy:.8f} Ha")
    print(f"  FCI configs: {qc_results.n_fci_configs}")
    print(f"  Max QSCI subspace: {conv_results.max_size}")

    return {
        'qc_results': qc_results,
        'conv_results': conv_results,
        'basis': basis,
        'n_orbitals': mol.nao,
    }


def build_h3_plus(bond_length: float, basis: str) -> gto.Mole:
    """
    Build H3+ triangular ion.

    Parameters
    ----------
    bond_length : float
        Bond length in Angstrom.
    basis : str
        Basis set name.

    Returns
    -------
    gto.Mole
        PySCF molecule object.
    """
    h = bond_length * np.sqrt(3) / 2
    coords = [
        (0.0 * bond_length, 0, 0),
        (1.0 * bond_length, 0, 0),
        (0.5 * bond_length, h, 0),
    ]
    geometry = '; '.join([f'H {x:.8f} {y:.8f} {z:.8f}' for x, y, z in coords])

    mol = gto.Mole()
    mol.build(
        atom=geometry,
        unit='Angstrom',
        basis=basis,
        charge=1,
        spin=0,
        verbose=0,
    )
    return mol


def sanitize_basis_name(basis: str) -> str:
    """Convert basis set name to filesystem-safe filename."""
    return basis.replace("*", "star").replace("+", "plus").replace("-", "_")


def save_individual_data(data_dir: Path, all_results: Dict):
    """
    Save convergence data for each basis set to individual CSV files.

    Parameters
    ----------
    data_dir : Path
        Directory to save data.
    all_results : dict
        Dictionary mapping basis set names to results.
    """
    for basis, results in all_results.items():
        basis_safe = sanitize_basis_name(basis)

        # Save convergence data
        conv_df = results['conv_results'].df.copy()
        conv_df.insert(0, 'basis', basis)
        conv_df.to_csv(data_dir / f'{basis_safe}_convergence.csv', index=False)

        # Save summary
        qc = results['qc_results']
        conv = results['conv_results']
        summary_data = {
            'basis': basis,
            'n_orbitals': results['n_orbitals'],
            'bond_length': BOND_LENGTH,
            'rhf_energy': qc.rhf.e_tot,
            'uhf_energy': qc.uhf.e_tot,
            'fci_energy': qc.fci_energy,
            'n_fci_configs': qc.n_fci_configs,
            'max_subspace_size': conv.max_size,
            'min_qsci_energy': conv.df['qsci_energy'].min(),
            'energy_diff_to_fci': conv.df['qsci_energy'].min() - qc.fci_energy,
            'n_configs_below_uhf': conv.n_configs_below_uhf if conv.n_configs_below_uhf else 'Never',
            'n_configs_reach_fci': conv.n_configs_reach_fci if conv.n_configs_reach_fci else 'Never',
        }
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(data_dir / f'{basis_safe}_summary.csv', index=False)

    # Save combined summary
    combined_summary = []
    for basis, results in all_results.items():
        qc = results['qc_results']
        conv = results['conv_results']
        combined_summary.append({
            'basis': basis,
            'n_orbitals': results['n_orbitals'],
            'rhf_energy': qc.rhf.e_tot,
            'uhf_energy': qc.uhf.e_tot,
            'fci_energy': qc.fci_energy,
            'n_fci_configs': qc.n_fci_configs,
            'max_subspace_size': conv.max_size,
            'min_qsci_energy': conv.df['qsci_energy'].min(),
            'energy_diff_to_fci': conv.df['qsci_energy'].min() - qc.fci_energy,
            'n_configs_below_uhf': conv.n_configs_below_uhf if conv.n_configs_below_uhf else 'Never',
            'n_configs_reach_fci': conv.n_configs_reach_fci if conv.n_configs_reach_fci else 'Never',
        })

    combined_df = pd.DataFrame(combined_summary)
    combined_df.to_csv(data_dir / 'all_basis_summary.csv', index=False)
    print(f"  Saved combined summary to all_basis_summary.csv")


def plot_basis_comparison(data_dir: Path, all_results: Dict):
    """
    Create comparison plot of QSCI convergence for all basis sets.

    Parameters
    ----------
    data_dir : Path
        Directory to save plots.
    all_results : dict
        Dictionary mapping basis set names to results.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 10))

    # Use a color palette
    colors = sns.color_palette("husl", len(all_results))

    # Plot each basis set
    for idx, (basis, results) in enumerate(all_results.items()):
        conv = results['conv_results']
        qc = results['qc_results']

        # Normalize energies relative to FCI for each basis
        energy_diff = conv.df['qsci_energy'] - qc.fci_energy

        ax.plot(
            conv.df['subspace_size'],
            energy_diff,
            'o-',
            label=f'{basis} (FCI: {qc.fci_energy:.6f} Ha)',
            linewidth=2,
            markersize=4,
            color=colors[idx],
            alpha=0.8
        )

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, label='FCI energy')

    ax.set_xlabel('Subspace Size (Number of Configurations)', fontsize=13)
    ax.set_ylabel('Energy - FCI Energy (Hartree)', fontsize=13)
    ax.set_title(
        f'H3+ QSCI Convergence: Basis Set Comparison\nBond Length = {BOND_LENGTH:.2f} Å',
        fontsize=15,
        fontweight='bold'
    )
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(data_dir / 'basis_comparison_convergence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved basis_comparison_convergence.png")

    # Also create a log-scale version for better visibility of small differences
    fig2, ax2 = plt.subplots(figsize=(14, 10))

    for idx, (basis, results) in enumerate(all_results.items()):
        conv = results['conv_results']
        qc = results['qc_results']

        # Use absolute value for log scale
        energy_diff = np.abs(conv.df['qsci_energy'] - qc.fci_energy)
        # Replace zeros with a small value
        energy_diff = np.where(energy_diff < 1e-12, 1e-12, energy_diff)

        ax2.semilogy(
            conv.df['subspace_size'],
            energy_diff,
            'o-',
            label=f'{basis}',
            linewidth=2,
            markersize=4,
            color=colors[idx],
            alpha=0.8
        )

    ax2.axhline(y=FCI_TOL, color='red', linestyle='--', linewidth=2,
                label=f'FCI tolerance ({FCI_TOL:.0e} Ha)')

    ax2.set_xlabel('Subspace Size (Number of Configurations)', fontsize=13)
    ax2.set_ylabel('|Energy - FCI Energy| (Hartree, log scale)', fontsize=13)
    ax2.set_title(
        f'H3+ QSCI Convergence: Basis Set Comparison (Log Scale)\nBond Length = {BOND_LENGTH:.2f} Å',
        fontsize=15,
        fontweight='bold'
    )
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(data_dir / 'basis_comparison_convergence_log.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved basis_comparison_convergence_log.png")


def plot_energy_levels_comparison(data_dir: Path, all_results: Dict):
    """
    Create bar chart comparing energy levels across basis sets.

    Parameters
    ----------
    data_dir : Path
        Directory to save plots.
    all_results : dict
        Dictionary mapping basis set names to results.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    basis_names = []
    rhf_energies = []
    uhf_energies = []
    fci_energies = []

    for basis, results in all_results.items():
        qc = results['qc_results']
        basis_names.append(basis)
        rhf_energies.append(qc.rhf.e_tot)
        uhf_energies.append(qc.uhf.e_tot)
        fci_energies.append(qc.fci_energy)

    x = np.arange(len(basis_names))
    width = 0.25

    ax.bar(x - width, rhf_energies, width, label='RHF', color='blue', alpha=0.8)
    ax.bar(x, uhf_energies, width, label='UHF', color='orange', alpha=0.8)
    ax.bar(x + width, fci_energies, width, label='FCI', color='green', alpha=0.8)

    ax.set_xlabel('Basis Set', fontsize=13)
    ax.set_ylabel('Energy (Hartree)', fontsize=13)
    ax.set_title(
        f'H3+ Energy Levels: Basis Set Comparison\nBond Length = {BOND_LENGTH:.2f} Å',
        fontsize=15,
        fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(basis_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(data_dir / 'basis_energy_levels.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved basis_energy_levels.png")


def print_summary_table(all_results: Dict):
    """
    Print a formatted summary table to console.

    Parameters
    ----------
    all_results : dict
        Dictionary mapping basis set names to results.
    """
    print("\nBasis Set Summary:")
    print("-" * 120)
    print(f"{'Basis':<10} {'N_orb':<7} {'RHF Energy':<14} {'UHF Energy':<14} {'FCI Energy':<14} "
          f"{'N_FCI':<7} {'Max_Sub':<8} {'Reach_FCI':<12}")
    print("-" * 120)

    for basis, results in all_results.items():
        qc = results['qc_results']
        conv = results['conv_results']
        reach_fci = conv.n_configs_reach_fci if conv.n_configs_reach_fci else 'Never'

        print(f"{basis:<10} {results['n_orbitals']:<7} {qc.rhf.e_tot:<14.8f} "
              f"{qc.uhf.e_tot:<14.8f} {qc.fci_energy:<14.8f} "
              f"{qc.n_fci_configs:<7} {conv.max_size:<8} {str(reach_fci):<12}")

    print("-" * 120)

    # Additional analysis
    print("\nConvergence Efficiency:")
    print("-" * 80)
    print(f"{'Basis':<10} {'Configs to reach FCI':<25} {'% of FCI space':<20}")
    print("-" * 80)

    for basis, results in all_results.items():
        conv = results['conv_results']
        qc = results['qc_results']

        if conv.n_configs_reach_fci:
            percentage = 100 * conv.n_configs_reach_fci / qc.n_fci_configs
            print(f"{basis:<10} {conv.n_configs_reach_fci:<25} {percentage:<20.2f}%")
        else:
            print(f"{basis:<10} {'Never reached':<25} {'N/A':<20}")

    print("-" * 80)


if __name__ == "__main__":
    main()

