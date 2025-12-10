"""
Analysis utilities for QSCI convergence and statevector handling.

This module extracts reusable functions from the research scripts so they can
be reused for arbitrary molecules/geometries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from math import log2
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyscf import gto, fci, scf
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from qiskit.quantum_info import Statevector

from sd_qsci.utils import uhf_from_rhf
from sd_qsci import circuit, hamiltonian, spin


# Defaults for tolerances (can be overridden by callers)
DEFAULT_SV_TOL = 1e-6
DEFAULT_FCI_TOL = 1e-6


@dataclass
class QuantumChemistryResults:
    """
    Container for quantum chemistry calculation results.

    Fields:
      mol: PySCF molecule
      rhf: RHF object
      uhf: UHF object
      sv: Statevector (from circuit.simulate)
      H: full Hamiltonian matrix (numpy array or sparse matrix)
      fci_energy: float
      n_fci_configs: int
      fci_vec: np.ndarray (full Fock-space FCI vector)
      bond_length: float | None
      spin_symm_amp: np.ndarray | None (spin-symmetric amplitudes)
    """
    mol: gto.Mole
    rhf: scf.RHF
    uhf: scf.UHF
    sv: Statevector
    H: np.ndarray
    fci_energy: float
    n_fci_configs: int
    fci_vec: np.ndarray
    bond_length: Optional[float] = None
    spin_symm_amp: Optional[np.ndarray] = None


@dataclass
class ConvergenceResults:
    """Container for convergence analysis results."""
    df: pd.DataFrame
    max_size: int
    n_configs_below_uhf: Optional[int]
    n_configs_reach_fci: Optional[int]
    n_configs_below_uhf_symm: Optional[int]
    n_configs_reach_fci_symm: Optional[int]


def calculate_convergence_data(
    qc_results: QuantumChemistryResults,
    sv_tol: float = DEFAULT_SV_TOL,
    fci_tol: float = DEFAULT_FCI_TOL,
    spin_symm: bool = False,
) -> ConvergenceResults:
    """
    Calculate QSCI and FCI subspace energies for varying subspace sizes.

    Parameters
    ----------
    qc_results : QuantumChemistryResults
        Quantum chemistry calculation results.
    sv_tol : float, optional
        Threshold for considering statevector amplitudes as present.
        Default is DEFAULT_SV_TOL.
    fci_tol : float, optional
        Tolerance for considering a QSCI energy equal to FCI.
        Default is DEFAULT_FCI_TOL.

    Returns
    -------
    ConvergenceResults
        Container with convergence analysis results.
    """
    max_idx = np.argwhere(np.abs(qc_results.sv.data) > sv_tol).ravel()
    max_size = len(max_idx)

    subspace_sizes = list(range(1, max_size + 1))
    qsci_energies = []
    symm_energies = []
    fci_subspace_energies = []
    mean_sample_numbers = []

    n_configs_below_uhf = None
    n_configs_reach_fci = None
    n_configs_below_uhf_symm = None
    n_configs_reach_fci_symm = None

    print("sv norm:", np.linalg.norm(qc_results.sv.data))
    print("spin symm amp norm:", np.linalg.norm(qc_results.spin_symm_amp))

    if spin_symm:
        data = qc_results.spin_symm_amp
    else:
        data = qc_results.sv.data

    for size in subspace_sizes:
        qsci_energy = calc_qsci_energy_with_size(qc_results.H, qc_results.sv.data, size)
        symm_energy = calc_qsci_energy_with_size(qc_results.H, qc_results.spin_symm_amp, size)
        qsci_energies.append(qsci_energy)
        symm_energies.append(symm_energy)

        fci_sub_energy = calc_fci_subspace_energy(qc_results.H, qc_results.fci_vec, size)
        fci_subspace_energies.append(fci_sub_energy)

        idx = np.argsort(np.abs(data))[-size:]
        min_coeff = np.min(np.abs(data[idx]))
        mean_sample_number = 1.0 / (min_coeff ** 2)
        mean_sample_numbers.append(mean_sample_number)

        if n_configs_below_uhf is None and qsci_energy < qc_results.uhf.e_tot:
            n_configs_below_uhf = size
        if n_configs_reach_fci is None and abs(qsci_energy - qc_results.fci_energy) < fci_tol:
            n_configs_reach_fci = size
        if n_configs_below_uhf_symm is None and symm_energy < qc_results.uhf.e_tot:
            n_configs_below_uhf_symm = size
        if n_configs_reach_fci_symm is None and abs(symm_energy - qc_results.fci_energy) < fci_tol:
            n_configs_reach_fci_symm = size

    df = pd.DataFrame({
        'subspace_size': subspace_sizes,
        'qsci_energy': qsci_energies,
        'spin_symm_energy': symm_energies,
        'fci_subspace_energy': fci_subspace_energies,
        'mean_sample_number': mean_sample_numbers
    })

    return ConvergenceResults(
        df=df,
        max_size=max_size,
        n_configs_below_uhf=n_configs_below_uhf,
        n_configs_reach_fci=n_configs_reach_fci,
        n_configs_below_uhf_symm=n_configs_below_uhf_symm,
        n_configs_reach_fci_symm=n_configs_reach_fci_symm,
    )


def calc_fci_energy(rhf, tol: float = 1e-10) -> tuple[float, int, np.ndarray]:
    """
    Compute FCI ground state and map PySCF's CI vector into the full Fock space.

    Parameters
    ----------
    rhf : scf.RHF
        Restricted Hartree-Fock object.
    tol : float, optional
        Tolerance for considering an FCI amplitude as nonzero. Default is 1e-10.

    Returns
    -------
    fci_energy : float
        FCI ground state energy.
    n_configs : int
        Number of significant FCI configurations.
    fci_vec : np.ndarray
        Full Fock-space FCI vector.
    """
    ci_solver = fci.FCI(rhf)
    fci_energy, fci_vec_ci = ci_solver.kernel()

    mol = rhf.mol
    nelec = mol.nelec

    fci_vec = fci_to_fock_space(fci_vec_ci, mol, nelec)
    n_configs = int(np.count_nonzero(np.abs(fci_vec) > tol))

    return fci_energy, n_configs, fci_vec


def calc_fci_subspace_energy(H, fci_vec, n_configs: int):
    """
    Energy of subspace spanned by the n_configs largest-amplitude FCI configurations.

    Parameters
    ----------
    H : np.ndarray or sparse matrix
        Full Hamiltonian matrix.
    fci_vec : np.ndarray
        Full Fock-space FCI vector.
    n_configs : int
        Number of configurations to include in the subspace.

    Returns
    -------
    float
        Ground state energy of the subspace.
    """
    idx = np.argsort(np.abs(fci_vec))[-n_configs:]
    H_sub = H[np.ix_(idx, idx)]

    if H_sub.shape[0] <= 2:
        eigenvalues = eigh(H_sub.toarray() if hasattr(H_sub, 'toarray') else H_sub, eigvals_only=True)
        E0 = float(np.min(eigenvalues))
    else:
        vals = eigsh(H_sub, k=1, which='SA', return_eigenvectors=False)
        E0 = float(vals[0])

    return E0


def calc_qsci_energy_with_size(
    H,
    data: np.ndarray,
    n_configs: int,
    return_vector: bool = False,
    spin_symmetry: bool = False,
):
    """
    Compute QSCI energy by diagonalising Hamiltonian in a subspace.

    Diagonalises the Hamiltonian restricted to the largest n_configs components
    of the provided statevector.

    Parameters
    ----------
    H : np.ndarray or sparse matrix
        Full Hamiltonian matrix.
    data : np.ndarray
        Quantum statevector data from circuit simulation.
    n_configs : int
        Number of configurations (largest amplitudes) to include in the subspace.
    return_vector : bool, optional
        If True, also return the full-space vector and configuration indices.
        Default is False.

    Returns
    -------
    float
        Ground state energy of the subspace.
    np.ndarray, optional
        Full-space statevector (returned only if return_vector is True).
    np.ndarray, optional
        Indices of configurations in the subspace (returned only if return_vector is True).
    """
    # n configs ordered by highest amplitude first
    idx = np.argsort(np.abs(data))[-n_configs:][::-1]

    if spin_symmetry:
        n_bits = int(log2(len(data)))
        idx = spin_symm_indices(idx, n_bits)[:n_configs]

    H_sub = H[np.ix_(idx, idx)]

    if H_sub.shape[0] <= 2:
        eigenvalues, eigenvectors = eigh(H_sub.toarray() if hasattr(H_sub, 'toarray') else H_sub)
        E0 = float(np.min(eigenvalues))
        psi0 = eigenvectors[:, 0]
    else:
        vals, vecs = eigsh(H_sub, k=1, which='SA')
        E0 = float(vals[0])
        psi0 = vecs[:, 0]

    if return_vector:
        psi0_full = np.zeros(data.shape, dtype=complex)
        psi0_full[idx] = psi0
        return E0, psi0_full, idx

    return E0


def fci_to_fock_space(fci_vec, mol: gto.Mole, nelec) -> np.ndarray:
    """
    Map PySCF's CI vector into the full Fock space vector.

    Converts PySCF's restricted CI vector to a full Fock-space representation
    using BLOCK spin ordering.

    Parameters
    ----------
    fci_vec : np.ndarray
        FCI vector from PySCF.
    mol : gto.Mole
        PySCF molecule object.
    nelec : tuple
        Tuple of (n_alpha, n_beta) electron counts.

    Returns
    -------
    np.ndarray
        Full Fock-space FCI vector.
    """
    from pyscf.fci import cistring

    nmo = mol.nao
    n_alpha, n_beta = nelec
    n_spin_orbitals = 2 * nmo

    alpha_strs = cistring.make_strings(range(nmo), n_alpha)
    beta_strs = cistring.make_strings(range(nmo), n_beta)

    fock_vec = np.zeros(2 ** n_spin_orbitals, dtype=complex)

    fci_vec_flat = fci_vec.flatten()

    for i_alpha, alpha_str in enumerate(alpha_strs):
        for i_beta, beta_str in enumerate(beta_strs):
            fock_idx = (alpha_str << nmo) | beta_str
            ci_idx = i_alpha * len(beta_strs) + i_beta
            fock_vec[fock_idx] = fci_vec_flat[ci_idx]

    return fock_vec


def run_quantum_chemistry_calculations(
        mol: gto.Mole,
        rhf: scf.RHF,
        bond_length: Optional[float],
) -> QuantumChemistryResults:
    """
    Run complete quantum chemistry calculations and circuit simulation.

    Performs UHF calculation, builds orbital-rotation circuit, simulates the
    statevector, and constructs the full Hamiltonian.

    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule object.
    rhf : scf.RHF
        Restricted Hartree-Fock object.
    bond_length : float, optional
        Bond length for reference. Default is None.

    Returns
    -------
    QuantumChemistryResults
        Container with all quantum chemistry calculation results.

    Raises
    ------
    RuntimeError
        If orbital rotation verification fails (statevector energy != UHF energy).
    """
    uhf = uhf_from_rhf(mol, rhf)
    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
    sv = circuit.simulate(qc)
    spin_symm_amp = spin_symm_amplitudes(sv.data)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)

    # Use np.vdot for robust dot product across array shapes
    sv_energy = np.vdot(sv.data, H.dot(sv.data)).real if hasattr(H, 'dot') else np.vdot(sv.data, H @ sv.data).real
    if not np.isclose(sv_energy, uhf.e_tot):
        raise RuntimeError("Orbital rotation verification failed: statevector energy != UHF energy")

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
        bond_length=bond_length,
        spin_symm_amp=spin_symm_amp,
    )


def plot_convergence_comparison(
    data_dir: Path,
    qc_results: QuantumChemistryResults,
    conv_results: ConvergenceResults,
    title_prefix: Optional[str] = None,
):
    """
    Create and save convergence comparison plot.

    Plots QSCI and FCI subspace energies against subspace size, comparing
    with RHF, UHF, and FCI reference energies.

    Parameters
    ----------
    data_dir : Path
        Directory to save the PNG file.
    qc_results : QuantumChemistryResults
        Quantum chemistry calculation results.
    conv_results : ConvergenceResults
        Convergence analysis results.
    title_prefix : str, optional
        Prefix to add to the plot title. Default is None.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(
        conv_results.df['subspace_size'],
        conv_results.df['qsci_energy'],
        'o-',
        label='QSCI (UHF-based selection)',
        linewidth=2,
        markersize=4,
        color='#0072B2',
    )

    ax.plot(
        conv_results.df['subspace_size'],
        conv_results.df['spin_symm_energy'],
        '^-',
        label='QSCI (spin recovery)',
        linewidth=2,
        markersize=4,
        color='#D55E00',
    )

    ax.plot(
        conv_results.df['subspace_size'],
        conv_results.df['fci_subspace_energy'],
        's-',
        label='FCI subspace (FCI-based selection)',
        linewidth=2,
        markersize=4,
        color='#009E73',
    )

    ax.axhline(y=qc_results.rhf.e_tot, color='blue', linestyle='--', linewidth=2,
               label=f'RHF: {qc_results.rhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.uhf.e_tot, color='orange', linestyle='--', linewidth=2,
               label=f'UHF: {qc_results.uhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.fci_energy, color='green', linestyle='--', linewidth=2,
               label=f'FCI: {qc_results.fci_energy:.6f} Ha')

    ax.set_xlabel('Subspace Size (Number of Configurations)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    bond_info = f"Bond Length = {qc_results.bond_length:.2f} Å" if qc_results.bond_length is not None else ""
    title = (f"Energy Convergence Comparison\n{bond_info}")
    if title_prefix:
        title = f"{title_prefix}: " + title
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    (Path(data_dir) / 'h6_qsci_convergence.png').parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(data_dir) / 'h6_qsci_convergence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_energy_vs_samples(
    data_dir: Path,
    qc_results: QuantumChemistryResults,
    conv_results: ConvergenceResults,
    title_prefix: Optional[str] = None,
):
    """
    Create and save energy vs mean-sample-number plot.

    Plots QSCI energy on a semilog scale against the mean sample number required
    for each subspace size.

    Parameters
    ----------
    data_dir : Path
        Directory to save the PNG file.
    qc_results : QuantumChemistryResults
        Quantum chemistry calculation results.
    conv_results : ConvergenceResults
        Convergence analysis results.
    title_prefix : str, optional
        Prefix to add to the plot title. Default is None.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.semilogx(conv_results.df['mean_sample_number'], conv_results.df['qsci_energy'], 'o-',
                label='QSCI (UHF-based selection)', linewidth=2, markersize=4, color='purple')

    ax.axhline(y=qc_results.rhf.e_tot, color='blue', linestyle='--', linewidth=2,
               label=f'RHF: {qc_results.rhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.uhf.e_tot, color='orange', linestyle='--', linewidth=2,
               label=f'UHF: {qc_results.uhf.e_tot:.6f} Ha')
    ax.axhline(y=qc_results.fci_energy, color='green', linestyle='--', linewidth=2,
               label=f'FCI: {qc_results.fci_energy:.6f} Ha')

    ax.set_xlabel('Mean Sample Number (log scale)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    bond_info = f"Bond Length = {qc_results.bond_length:.2f} Å" if qc_results.bond_length is not None else ""
    title = (f"Energy vs Mean Sample Number\n{bond_info}")
    if title_prefix:
        title = f"{title_prefix}: " + title
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    (Path(data_dir) / 'h6_energy_vs_samples.png').parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(data_dir) / 'h6_energy_vs_samples.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_statevector_coefficients(
    qsci_vec: np.ndarray,
    fci_vec: np.ndarray,
    data_dir: Path,
    n_top: int = 20,
):
    """
    Plot comparison of QSCI and FCI statevector coefficients.

    Creates two plots: one showing the top n_top configurations as a bar chart,
    and another showing all significant configurations on a log scale.

    Parameters
    ----------
    qsci_vec : np.ndarray
        QSCI statevector.
    fci_vec : np.ndarray
        FCI statevector.
    data_dir : Path
        Directory to save the PNG files.
    n_top : int, optional
        Number of top configurations to show in the first plot. Default is 20.

    Returns
    -------
    dict
        Dictionary with statistics:
        - 'n_significant_fci': Number of significant FCI configurations
        - 'n_significant_qsci': Number of significant QSCI configurations
        - 'max_fci_coef': Maximum FCI coefficient magnitude
        - 'max_qsci_coef': Maximum QSCI coefficient magnitude
        - 'overlap': Overlap between FCI and QSCI vectors
    """
    fci_abs = np.abs(fci_vec)
    top_indices = np.argsort(fci_abs)[-n_top:][::-1]

    qsci_coefs = np.abs(qsci_vec[top_indices])
    fci_coefs = fci_abs[top_indices]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(n_top)
    width = 0.35

    ax.bar(x - width/2, fci_coefs, width, label='FCI', color='green', alpha=0.8)
    ax.bar(x + width/2, qsci_coefs, width, label='QSCI', color='purple', alpha=0.8)

    ax.set_xlabel('Configuration Index (sorted by FCI amplitude)', fontsize=12)
    ax.set_ylabel('|Coefficient|', fontsize=12)
    ax.set_title(f'Top {n_top} Configuration Coefficients', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in top_indices], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    (Path(data_dir) / 'statevector_coefficients.png').parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(data_dir) / 'statevector_coefficients.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(14, 8))

    significant_mask = (fci_abs > 1e-10) | (np.abs(qsci_vec) > 1e-10)
    significant_indices = np.where(significant_mask)[0]

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
    (Path(data_dir) / 'statevector_coefficients_full.png').parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(data_dir) / 'statevector_coefficients_full.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # Return some stats for programmatic use
    stats = {
        'n_significant_fci': int(np.sum(fci_abs > 1e-10)),
        'n_significant_qsci': int(np.sum(np.abs(qsci_vec) > 1e-10)),
        'max_fci_coef': float(np.max(fci_abs)),
        'max_qsci_coef': float(np.max(np.abs(qsci_vec))),
        'overlap': float(np.abs(np.vdot(fci_vec, qsci_vec)))
    }
    return stats


def save_convergence_data(
    data_dir: Path,
    qc_results: QuantumChemistryResults,
    conv_results: ConvergenceResults,
):
    """
    Save convergence data and summary to CSV files.

    Saves the convergence dataframe to 'h6_qsci_convergence.csv' and a
    summary of key quantities to 'h6_summary.csv'.

    Parameters
    ----------
    data_dir : Path
        Directory to save the CSV files.
    qc_results : QuantumChemistryResults
        Quantum chemistry calculation results.
    conv_results : ConvergenceResults
        Convergence analysis results.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    conv_results.df.to_csv(Path(data_dir) / 'h6_qsci_convergence.csv', index=False)

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
    summary_df.to_csv(Path(data_dir) / 'h6_summary.csv', index=False)


def setup_data_directory(base: Optional[Path] = None) -> Path:
    """
    Create and return a data directory.

    Creates a data directory adjacent to this module, or under the specified
    base directory if provided.

    Parameters
    ----------
    base : Path, optional
        Base directory for data storage. If None, uses a directory relative
        to this module. Default is None.

    Returns
    -------
    Path
        Path to the data directory (created if it doesn't exist).
    """
    if base is None:
        data_dir = Path(__file__).parent.parent / 'research_data' / Path(__file__).stem
    else:
        data_dir = Path(base)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def spin_symm_indices(idx, n_bits):
    new_indices = []
    bitstring = bin(idx)[2:].zfill(n_bits)
    bitstrings = spin.spin_symmetric_configs(bitstring)
    for bs in bitstrings:
        new_indices.append(int(bs, 2))
    return new_indices


def spin_symm_amplitudes(sv_data: np.ndarray) -> np.ndarray:
    """
    Make amplitudes equal to largest for all spin-symmetric configurations.
    """
    sv_data_new = sv_data.copy()
    calculated_indices = set()
    indices = np.argsort(np.abs(sv_data_new))[::-1]
    n_bits = int(np.log2(sv_data_new.size))
    for i in indices:
        if i not in calculated_indices:
            symm_indices = spin_symm_indices(i, n_bits)
            for j in symm_indices:
                calculated_indices.add(j)
                sv_data_new[j] = sv_data[i]
    return sv_data_new


__all__ = [
    'QuantumChemistryResults',
    'ConvergenceResults',
    'calculate_convergence_data',
    'calc_fci_energy',
    'calc_fci_subspace_energy',
    'calc_qsci_energy_with_size',
    'fci_to_fock_space',
    'run_quantum_chemistry_calculations',
    'plot_convergence_comparison',
    'plot_energy_vs_samples',
    'plot_statevector_coefficients',
    'save_convergence_data',
    'setup_data_directory',
]
