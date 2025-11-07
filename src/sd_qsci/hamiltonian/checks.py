"""
Check molecular orbital integrals give correct RHF energy.
"""
import numpy as np
from pyscf import ao2mo

def rhf_energy_from_mo_integrals(mol, rhf) -> float:
    """
    Compute RHF total energy directly from MO integrals. Assumes closed-shell
    (RHF) occupations: doubly-occupy the lowest nocc spatial MOs.
    """
    C = rhf.mo_coeff
    nmo = C.shape[1]
    nocc = rhf.mol.nelectron // 2  # number of doubly occupied spatial MOs

    # 1e: AO -> MO (spatial)
    h_ao = rhf.get_hcore()
    h_mo = C.T @ h_ao @ C

    # 2e: AO -> MO (chemist), then restore and convert to physicist: (pq|rs) -> (pr|qs)
    eri_chem = ao2mo.full(mol, C)  # packed chemist
    eri_chem = ao2mo.restore(1, eri_chem, nmo)  # (nmo,nmo,nmo,nmo) chemist
    g_phys = np.transpose(eri_chem, (0, 2, 1, 3))  # physicist (pr|qs)

    # RHF formula in *spatial* MO basis
    occ = range(nocc)
    e_one = 2.0 * sum(h_mo[i, i] for i in occ)
    e_coul = sum(2.0 * g_phys[i, j, i, j] for i in occ for j in occ)   # 2*(ij|ij)
    e_exch = sum(      g_phys[i, j, j, i] for i in occ for j in occ)   #   (ij|ji)
    e_elec = e_one + (e_coul - e_exch)
    return e_elec + mol.energy_nuc()
