"""
Profile the performance of RHF to UHF workflow.
"""

import numpy as np
from line_profiler import profile
from pyscf import gto, scf

from sd_qsci import circuit, energy, hamiltonian, utils


@profile
def main():
    mol = h6_triangular_lattice(bond_length=2)
    rhf = scf.RHF(mol).run()
    uhf = utils.uhf_from_rhf(mol, rhf)
    qc = circuit.rhf_uhf_orbital_rotation_circuit(mol, rhf, uhf)
    sv = circuit.simulate(qc)
    H = hamiltonian.hamiltonian_from_pyscf(mol, rhf)
    sv_energy = (sv.data.conj().T @ H @ sv.data).real
    assert np.isclose(sv_energy, uhf.e_tot)

    fci_energy = energy.fci_energy(rhf)
    qsci_energy, qsci_idx = energy.qsci_energy(H, sv)


def h6_triangular_lattice(bond_length):
    """
    Build a triangular lattice of 6 hydrogen atoms.

    Parameters
    ----------
    bond_length : float
        Distance between adjacent hydrogen atoms in Angstroms.

    Returns
    -------
    mol : gto.Mole
    """
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


if __name__ == "__main__":
    main()
