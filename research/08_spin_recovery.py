"""
H6 Chain Spin Recovery Analysis
================================================

This script performs a comprehensive analysis of Quantum Selected Configuration
Interaction (QSCI) convergence for a 6-hydrogen chain.

Overview
--------
For a fixed bond length (e.g. 2.0 A), the script:
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
    - Diagonalise to get ground state energy in the subspace
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
from sd_qsci.utils import uhf_from_rhf
from sd_qsci import circuit, hamiltonian, spin


# Script-specific tolerances
SV_TOL = 1e-2
FCI_TOL = 1e-6


def main():
    """
    Run H6 chain energy calculations and analyze QSCI convergence.
    """
    # Setup
    bond_length = 2.0
    n_atoms = 6
    print(f"Running H{n_atoms} chain bond length: {bond_length:.2f} Angstrom")

    run_full_analysis(bond_length, n_atoms)
    # print_top_configs(bond_length, n_atoms)


def run_full_analysis(bond_length, n_atoms):
    filename = Path(__file__).stem
    data_dir = Path(__file__).parent / 'data' / filename / f"bond_length_{bond_length:.2f}_spin_symm"

    # Run quantum chemistry calculations
    mol = build_h_chain(bond_length, n_atoms)
    rhf = scf.RHF(mol).run()
    qc_results = analysis.run_quantum_chemistry_calculations(mol, rhf, bond_length)

    # Calculate convergence data
    conv_results = analysis.calculate_convergence_data(qc_results, spin_symm=True)

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


def print_top_configs(bond_length=2, n_atoms=6):
    """
    Printing the top amplitudes from the rotated UHF statevector.
    This is to quickly see if the spin recovery is required/working.
    """
    # Run quantum chemistry calculations
    mol = build_h_chain(bond_length, n_atoms)
    rhf = scf.RHF(mol).run()

    uhf = uhf_from_rhf(mol, rhf)
    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
    sv = circuit.simulate(qc)
    print("sv norm:", np.linalg.norm(sv.data))

    sorted_idx = np.argsort(np.abs(sv.data))[::-1]
    max_idx = sorted_idx[:20]
    n_bits = int(np.log2(sv.data.size))

    # For the highest amplitude configurations print their amplitude, index,
    # bitstring, and occupation vector in a table format.
    print("Norm" + ' '*6 + "Int" + ' '*5 + "Bitstring" + ' '*(n_bits-4) + "Occ")
    for i in max_idx:
        bitstring = bin(i)[2:].zfill(n_bits)
        occ_vec = occupation_vector(bitstring, n_bits)
        sv_amp = np.abs(sv.data[i])
        bitstring_ab = bitstring[:n_bits//2] + ' ' + bitstring[n_bits//2:]
        print(f"{sv_amp:.4f}    {i:4d}    {bitstring_ab}    {occ_vec}")
    print('\n')

    symm_amp = analysis.spin_symm_amplitudes(sv.data)
    print("symm amp norm:", np.linalg.norm(symm_amp))

    sorted_idx = np.argsort(np.abs(symm_amp))[::-1]
    max_idx = sorted_idx[:20]
    n_bits = int(np.log2(symm_amp.size))

    # For the highest amplitude configurations after spin recovery, print their
    # amplitude, index, bitstring, and occupation vector in a table format.
    print("Norm" + ' '*6 + "Int" + ' '*5 + "Bitstring" + ' '*(n_bits-4) + "Occ")
    for i in max_idx:
        bitstring = bin(i)[2:].zfill(n_bits)
        occ_vec = occupation_vector(bitstring, n_bits)
        sv_amp = np.abs(symm_amp[i])
        bitstring_ab = bitstring[:n_bits//2] + ' ' + bitstring[n_bits//2:]
        print(f"{sv_amp:.4f}    {i:4d}    {bitstring_ab}    {occ_vec}")


def occupation_vector(bitstring, n_bits) -> str:
    """
    Given a bitstring, return the occupation vector.
    Closed shells are represented by 0 or 2 for unoccupied or occupied.
    Open shells are represented by α or β for up or down spin occupancy.

    The RHF molecular orbitals are ordered from highest to lowest energy.
    This is inline with the Qiskit qubit ordering convention.

    Parameters
    ----------
    bitstring: str
        Binary representation of the bitstring.

    Returns
    -------
    occ_vec: str
        Occupation vector representation of the bitstring.
    """
    # Ensure string format
    bitstring = str(bitstring)
    alpha, beta = bitstring[:n_bits//2], bitstring[n_bits//2:]
    occ_vec = ""
    for i in range(n_bits//2):
        if alpha[i] == "1" and beta[i] == "1":
            occ_vec += '2'
        elif alpha[i] == "1":
            occ_vec += '\u03b1'
        elif beta[i] == "1":
            occ_vec += '\u03b2'
        else:
            occ_vec += '0'
    return occ_vec


def build_h_chain(bond_length, n_atoms=6) -> gto.Mole:
    """
    Build a chain of hydrogen atoms.
    """
    coords = [(i * bond_length, 0, 0) for i in range(n_atoms)]
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


def print_summary(
        data_dir: Path,
        qc_results: analysis.QuantumChemistryResults,
        conv_results: analysis.ConvergenceResults,
        qsci_energy_final: float,
) -> None:
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
