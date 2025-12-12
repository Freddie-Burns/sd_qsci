"""
H3+ Triangular Hydrogen Ion - Quick Basis Set Comparison Test.

This is a lighter version that tests fewer basis sets for quick validation.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, scf

from sd_qsci import analysis


# Configuration - smaller set for testing
BASIS_SETS = ["sto-3g", "3-21g", "6-31g*", "cc-pvdz"]
BOND_LENGTH = 2.0
SV_TOL = 1e-12
FCI_TOL = 1e-6


def main():
    """
    Run H3+ calculations for a few basis sets (quick test).
    """
    script_name = Path(__file__).stem
    data_dir = Path(__file__).parent / 'data' / script_name
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"H3+ Basis Set Comparison Study (Quick Test)")
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

            all_results[basis] = {
                'qc_results': qc_results,
                'conv_results': conv_results,
                'basis': basis,
                'n_orbitals': mol.nao,
            }
            print(f"✓ Successfully completed {basis}")

        except Exception as e:
            print(f"✗ Error with {basis}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("\nNo successful calculations. Exiting.")
        return

    # Create comparison plot
    print(f"\n{'=' * 60}")
    print("Creating comparison plot...")
    print(f"{'=' * 60}")
    plot_basis_comparison(data_dir, all_results)

    # Save combined CSV
    save_combined_data(data_dir, all_results)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print_summary_table(all_results)

    print(f"\n{'=' * 60}")
    print(f"Analysis complete! Results saved to:")
    print(f"  {data_dir}")
    print(f"{'=' * 60}")


def build_h3_plus(bond_length: float, basis: str) -> gto.Mole:
    """Build H3+ triangular ion."""
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


def plot_basis_comparison(data_dir: Path, all_results: Dict):
    """Create comparison plot of QSCI convergence for all basis sets."""
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Use a color palette
    colors = sns.color_palette("husl", len(all_results))

    # Top plot: Energy difference from FCI (linear scale)
    for idx, (basis, results) in enumerate(all_results.items()):
        conv = results['conv_results']
        qc = results['qc_results']

        energy_diff = conv.df['qsci_energy'] - qc.fci_energy

        ax1.plot(
            conv.df['subspace_size'],
            energy_diff,
            'o-',
            label=f'{basis} (FCI: {qc.fci_energy:.6f} Ha)',
            linewidth=2,
            markersize=4,
            color=colors[idx],
            alpha=0.8
        )

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='FCI energy')
    ax1.set_xlabel('Subspace Size (Number of Configurations)', fontsize=12)
    ax1.set_ylabel('Energy - FCI Energy (Hartree)', fontsize=12)
    ax1.set_title(
        f'H3+ QSCI Convergence: Basis Set Comparison (Linear)\nBond Length = {BOND_LENGTH:.2f} Å',
        fontsize=14,
        fontweight='bold'
    )
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Absolute energy difference (log scale)
    for idx, (basis, results) in enumerate(all_results.items()):
        conv = results['conv_results']
        qc = results['qc_results']

        energy_diff = np.abs(conv.df['qsci_energy'] - qc.fci_energy)
        # Replace zeros with a small value for log scale
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
    ax2.set_xlabel('Subspace Size (Number of Configurations)', fontsize=12)
    ax2.set_ylabel('|Energy - FCI Energy| (Hartree, log scale)', fontsize=12)
    ax2.set_title(
        f'H3+ QSCI Convergence: Basis Set Comparison (Log Scale)\nBond Length = {BOND_LENGTH:.2f} Å',
        fontsize=14,
        fontweight='bold'
    )
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(data_dir / 'basis_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved basis_comparison.png")


def sanitize_basis_name(basis: str) -> str:
    """Convert basis set name to filesystem-safe filename."""
    return basis.replace("*", "star").replace("+", "plus").replace("-", "_")


def save_combined_data(data_dir: Path, all_results: Dict):
    """Save combined convergence data to CSV."""
    # Combine all convergence data
    combined_rows = []
    for basis, results in all_results.items():
        conv_df = results['conv_results'].df.copy()
        conv_df.insert(0, 'basis', basis)
        conv_df.insert(1, 'fci_energy', results['qc_results'].fci_energy)
        combined_rows.append(conv_df)

    combined_df = pd.concat(combined_rows, ignore_index=True)
    combined_df.to_csv(data_dir / 'all_basis_convergence.csv', index=False)
    print(f"  Saved all_basis_convergence.csv")

    # Summary table
    summary_rows = []
    for basis, results in all_results.items():
        qc = results['qc_results']
        conv = results['conv_results']
        summary_rows.append({
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

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(data_dir / 'all_basis_summary.csv', index=False)
    print(f"  Saved all_basis_summary.csv")


def print_summary_table(all_results: Dict):
    """Print a formatted summary table to console."""
    print("\nBasis Set Summary:")
    print("-" * 110)
    print(f"{'Basis':<10} {'N_orb':<7} {'RHF Energy':<14} {'FCI Energy':<14} "
          f"{'N_FCI':<7} {'Max_Sub':<8} {'Reach_FCI':<12}")
    print("-" * 110)

    for basis, results in all_results.items():
        qc = results['qc_results']
        conv = results['conv_results']
        reach_fci = conv.n_configs_reach_fci if conv.n_configs_reach_fci else 'Never'

        print(f"{basis:<10} {results['n_orbitals']:<7} {qc.rhf.e_tot:<14.8f} "
              f"{qc.fci_energy:<14.8f} "
              f"{qc.n_fci_configs:<7} {conv.max_size:<8} {str(reach_fci):<12}")

    print("-" * 110)

    # Convergence efficiency
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
