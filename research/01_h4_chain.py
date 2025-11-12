"""
Create H4 chain and compare energies and spins from RHF, UHF, FCI, and QSCI.
Ensure the statevector after orbital rotation gives the same energy and spin
as the input UHF state.
"""

import numpy as np
import pandas as pd
from pyscf import gto, scf, fci
from scipy.sparse.linalg import eigsh

from sd_qsci.utils import uhf_from_rhf, uhf_to_rhf_unitaries
from sd_qsci import circuit, hamiltonian, spin, utils


def main():
    for bond_length in [2]:
        mol = build_h4_chain(bond_length)
        rhf = scf.RHF(mol).run()
        uhf = uhf_from_rhf(mol, rhf)
        qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
        sv = circuit.run_statevector(qc)
        H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)
        qsci_energy(H, sv)


def build_h4_chain(bond_length):
    """
    Build a linear chain of 4 hydrogen atoms with equal spacing.

    Creates a PySCF molecule object for a linear H4 chain with atoms
    positioned along the x-axis at x = 0, bond_length, 2*bond_length,
    and 3*bond_length. The molecule is constructed with STO-3G basis set,
    neutral charge, and singlet spin state.

    Parameters
    ----------
    bond_length : float
        Distance between adjacent hydrogen atoms in Angstroms.

    Returns
    -------
    mol : gto.Mole
        PySCF molecule object representing the H4 chain with the
        following properties:
        - basis: 'sto-3g'
        - charge: 0
        - spin: 0 (singlet)
        - 4 hydrogen atoms in linear configuration
    """
    coords = [
        (0 * bond_length, 0, 0),
        (1 * bond_length, 0, 0),
        (2 * bond_length, 0, 0),
        (3 * bond_length, 0, 0),
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


def qsci_energy(H, statevector):
    sampled_configs = np.argwhere(np.abs(statevector.data) > 1e-12)[:,1]
    print(sampled_configs)
    return (statevector.data.conj() @ H @ statevector.data).real


def analyse_statevector(mol, rhf, uhf, statevector):
    # Analyze statevector
    sv_abs = np.abs(statevector.data)
    idx_sorted = np.argsort(sv_abs)[::-1]
    sv_abs = sv_abs[idx_sorted]

    bitstring_ints = np.arange(len(sv_abs))[idx_sorted]
    bitstrings = [f"{x:0{2 * mol.nao}b}" for x in bitstring_ints]

    df = pd.DataFrame({'Bitstring': bitstrings, 'Psi abs': sv_abs})
    df = df.round(2)
    print(df)

    # Energy comparison
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)
    sv_energy = (statevector.data.conj() @ H @ statevector.data).real
    uhf_energy = uhf.e_tot
    rhf_energy = rhf.e_tot

    ci_solver = fci.FCI(rhf)
    fci_energy, fci_vec = ci_solver.kernel()
    fci_s2, mult = ci_solver.spin_square(fci_vec, mol.nao, mol.nelec)

    print("FCI energy:", fci_energy)
    print("RHF energy:", rhf_energy)
    print("UHF energy:", uhf_energy)
    print("Statevector energy (expectation):", sv_energy)

    # Spin comparison
    idx = np.argwhere(np.abs(statevector.data) > 1e-12).ravel()
    H_sub = H[np.ix_(idx, idx)]
    E0, psi0 = eigsh(H_sub, k=1, which='SA')

    S2_fermi = spin.total_spin_S2(mol.nao)
    S2_sub = S2_fermi[np.ix_(idx, idx)]
    qsci_s2 = (psi0.conj().T @ S2_sub @ psi0).real

    uhf_s2, uhf_multiplicity = uhf.spin_square()
    rhf_bitstring = int(f"{'0' * mol.nelectron}{'1' * (2 * mol.nao - mol.nelectron)}", 2)
    rhf_s2 = S2_fermi[rhf_bitstring, rhf_bitstring].real

    print("FCI spin:", fci_s2)
    print("RHF spin:", rhf_s2)
    print("UHF spin:", uhf_s2)
    print("QSCI spin:", qsci_s2[0, 0].real if qsci_s2.ndim > 1 else qsci_s2.real)

    # Check spin symmetry
    sampled_configs, symm_configs = utils.find_spin_symmetric_configs(n_bits=2 * mol.nao, idx=idx)
    print("Spin symmetry preserved:", np.array_equal(sampled_configs, symm_configs))

    idx_symm = [int(x, 2) for x in symm_configs]
    idx_symm = sorted(set(idx_symm))
    H_sub_symm = H[np.ix_(idx_symm, idx_symm)]
    E0_symm, psi0_symm = eigsh(H_sub_symm, k=1, which='SA')

    print("Configs with symmetry:", len(idx_symm))
    print("QSCI energy (symmetric):", E0_symm[0])
