"""

"""

from pathlib import Path

from pyscf import gto, scf
from pyscf.cc import CCSD
from qiskit_aer import Aer
from qiskit.visualization import circuit_drawer

from sd_qsci import analysis, circuit, hamiltonian
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
    data_dir = Path(__file__).parent / 'data' / filename / f"bond_length_{bond_length:.2f}"

    print(f"Running H{n_atoms} chain bond length: {bond_length:.2f} Angstrom")

    # Run quantum chemistry calculations
    mol = build_h_chain(bond_length, n_atoms)
    rhf = scf.RHF(mol).run()
    uhf = uhf_from_rhf(mol, rhf)
    ccsd = CCSD(rhf).run()

    backend = Aer.get_backend("statevector_simulator")
    qc = circuit.get_lucj_circuit(ccsd_obj=ccsd, backend=backend, n_reps=10)

    # Ensure output directory exists and save an image of the LUCJ circuit
    data_dir.mkdir(parents=True, exist_ok=True)
    circuit_path = data_dir / 'lucj_circuit.png'
    try:
        circuit_drawer(qc, output='mpl', filename=str(circuit_path), fold=-1, idle_wires=False)
        print(f"Saved LUCJ circuit image to: {circuit_path}")
    except Exception as e:
        # Fall back silently if visualization is unavailable
        print(f"Warning: Failed to save LUCJ circuit image ({e})")

    sv = circuit.simulate(qc)
    spin_symm_amp = analysis.spin_symm_amplitudes(sv.data)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)

    fci_energy, n_fci_configs, fci_vec = analysis.calc_fci_energy(rhf)

    qc_results = analysis.QuantumChemistryResults(
        mol=mol,
        rhf=rhf,
        uhf=uhf,
        sv=sv,
        H=H,
        fci_energy=fci_energy,
        n_fci_configs=n_fci_configs,
        fci_vec=fci_vec,
        bond_length=bond_length,
        spin_symm_amp=spin_symm_amp,
    )

    # Calculate convergence data
    conv_results = analysis.calc_convergence_data(qc_results, spin_symm=True)

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

    print_summary(data_dir, qc_results, conv_results, qsci_energy_final)


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
    print("  - lucj_circuit.png (LUCJ circuit diagram)")


if __name__ == "__main__":
    main()
