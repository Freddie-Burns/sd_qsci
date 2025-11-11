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
from sd_qsci import circuit
from sd_qsci.hamiltonian import hamiltonian_from_pyscf
from sd_qsci.spin import total_spin_S2
from sd_qsci.utils import find_spin_symmetric_configs

# Setup molecule
a = 2  # Angstrom
coords = [
    (0 * a, 0, 0),
    (1 * a, 0, 0),
    (2 * a, 0, 0),
    (3 * a, 0, 0),
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

# Prepare RHF and UHF
rhf = scf.RHF(mol)
rhf.run()
uhf = uhf_from_rhf(mol, rhf)
Ua, Ub = uhf_to_rhf_unitaries(mol, rhf, uhf)

# Create and run quantum circuit
# qc = circuit.orbital_rotation_circuit(nao=mol.nao, nelec=mol.nelec, Ua=Ua, Ub=Ub)
qc = circuit.rhf_uhf_orbital_rotation_circuit(mol=mol, rhf=rhf, uhf=uhf)
statevector = circuit.run_statevector(qc)

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
H = hamiltonian_from_pyscf(mol, rhf)
qsci_energy = (statevector.data.conj() @ (H @ statevector.data)).real
uhf_energy = uhf.e_tot
rhf_energy = rhf.e_tot

ci_solver = fci.FCI(rhf)
fci_energy, fci_vec = ci_solver.kernel()
fci_s2, mult = ci_solver.spin_square(fci_vec, mol.nao, mol.nelec)

print("FCI energy:", fci_energy)
print("RHF energy:", rhf_energy)
print("UHF energy:", uhf_energy)
print("QSCI energy (expectation):", qsci_energy)

# Spin comparison
idx = np.argwhere(np.abs(statevector.data) > 1e-12).ravel()
H_sub = H[np.ix_(idx, idx)]
E0, psi0 = eigsh(H_sub, k=1, which='SA')

S2_fermi = total_spin_S2(mol.nao)
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
sampled_configs, symm_configs = find_spin_symmetric_configs(n_bits=2 * mol.nao, idx=idx)
print("Spin symmetry preserved:", np.array_equal(sampled_configs, symm_configs))

idx_symm = [int(x, 2) for x in symm_configs]
idx_symm = sorted(set(idx_symm))
H_sub_symm = H[np.ix_(idx_symm, idx_symm)]
E0_symm, psi0_symm = eigsh(H_sub_symm, k=1, which='SA')

print("Configs with symmetry:", len(idx_symm))
print("QSCI energy (symmetric):", E0_symm[0])
