"""

"""

from pathlib import Path

import numpy as np
from pyscf import gto, scf

from sd_qsci import analysis, circuit
from sd_qsci.utils import uhf_from_rhf


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

    filename = Path(__file__).stem
    data_dir = Path(__file__).parent / 'data' / filename / f"bond_length_{bond_length:.2f}_spin_symm"

    print(f"Running H{n_atoms} chain bond length: {bond_length:.2f} Angstrom")

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
    analysis.plot_convergence_comparison(data_dir, qc_results, conv_results, ylog=True)

    # Compute final QSCI wavefunction and plot coefficients
    print(f"\nComputing QSCI ground state wavefunction with {conv_results.max_size} configurations...")
    qsci_energy_final, qsci_vec, qsci_indices = analysis.calc_qsci_energy_with_size(
        qc_results.H,
        qc_results.sv,
        conv_results.max_size, return_vector=True,
    )

    analysis.plot_statevector_coefficients(
        qc_results.sv.data,
        qc_results.fci_vec,
        data_dir,
        n_top=20,
    )
    analysis.plot_total_spin_vs_subspace(
        data_dir=data_dir,
        qc_results=qc_results,
        conv_results=conv_results,
        title_prefix="H6 Chain"
    )


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