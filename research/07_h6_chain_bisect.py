"""
H6 Chain, two separated equilibrium 3 atom chains
"""

from pathlib import Path

import numpy as np
from pyscf import gto, scf

from sd_qsci import analysis

# Script-specific tolerances
SV_TOL = 1e-2
FCI_TOL = 1e-6


def main():
    """
    Run H6 triangular lattice energy calculations and analyze QSCI convergence.
    """
    # Setup
    filename = Path(__file__).stem
    data_dir = Path(__file__).parent / 'data' / filename

    # Run quantum chemistry calculations
    mol = build_h6_chain()
    rhf = scf.RHF(mol).run()
    qc_results = analysis.run_quantum_chemistry_calculations(mol, rhf, bond_length=None)

    # Calculate convergence data
    conv_results = analysis.calc_convergence_data(qc_results)

    # Save data to CSV
    analysis.save_convergence_data(data_dir, qc_results, conv_results)

    # Create plots
    analysis.plot_energy_vs_samples(data_dir, qc_results, conv_results)
    analysis.plot_convergence_comparison(data_dir, qc_results, conv_results)

    # Compute final QSCI wavefunction and plot coefficients
    print(f"\nComputing QSCI ground state wavefunction with {conv_results.max_size} configurations...")
    qsci_energy_final, qsci_vec, qsci_indices = analysis.calc_qsci_energy_with_size(
        qc_results.H, qc_results.sv, conv_results.max_size, return_vector=True)
    analysis.plot_statevector_coefficients(qc_results.sv.data, qc_results.fci_vec, data_dir, n_top=20)

    # Print summary
    print_summary(data_dir, qc_results, conv_results, qsci_energy_final)


def build_h6_chain():
    """
    Build a triangular lattice of 6 hydrogen atoms.
    """
    coords = [
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (4, 0, 0),
        (5, 0, 0),
        (6, 0, 0),
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


def print_summary(data_dir: Path, qc_results: analysis.QuantumChemistryResults,
                 conv_results: analysis.ConvergenceResults, qsci_energy_final: float):
    """
    Print summary of results to console.
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


if __name__ == "__main__":
    main()
