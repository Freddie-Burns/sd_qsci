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
3. Analyses how QSCI energy converges to FCI by varying the number of
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
    bond_length = 2.0
    data_dir = Path(__file__).parent / 'data' / '03c_h6_chain' / f"bond_length_{bond_length:.2f}"
    print(f"Running bond length: {bond_length:.2f} Angstrom")

    # Run quantum chemistry calculations
    mol = build_h6_lattice(bond_length)
    rhf = scf.RHF(mol).run()
    qc_results = analysis.run_quantum_chemistry_calculations(mol, rhf, bond_length)

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


def build_h6_lattice(bond_length):
    """
    Build a triangular lattice of 6 hydrogen atoms.
    """
    # Create triangular lattice: 3 atoms in first row, 3 in second row
    h = bond_length * np.sqrt(3) / 2
    coords = [
        (0.0 * bond_length, 0, 0),
        (1.0 * bond_length, 0, 0),
        (2.0 * bond_length, 0, 0),
        (3.0 * bond_length, 0, 0),
        (4.0 * bond_length, 0, 0),
        (5.0 * bond_length, 0, 0),
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
