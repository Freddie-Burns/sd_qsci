import numpy as np
from line_profiler import profile
from pyscf import ao2mo
from .spin_blocks import spin_expand_1e, spin_expand_2e_phys
from .fermion_hamiltonian import hamiltonian_matrix


@profile
def hamiltonian_from_pyscf(mol, rhf):
    """
    Return the sparse Hamiltonian (csr) on the full spin-orbital Fock space.
    Acts on an occupation number vector basis.
    BLOCK spin ordering: [α0..α(n-1), β0..β(n-1)].
    """
    C = rhf.mo_coeff
    nmo = C.shape[1]

    # 1e AO->MO
    h1_ao = rhf.get_hcore()
    h1_spatial = C.T @ h1_ao @ C

    # 2e AO->MO, chemist order -> restore -> physicist order (swap middle indices)
    eri_mo_packed = ao2mo.full(mol, C)  # packed (pq|rs), chemist
    eri_mo_chem   = ao2mo.restore(1, eri_mo_packed, nmo)  # (n,n,n,n)
    g_spatial_phys = np.transpose(eri_mo_chem, (0, 2, 1, 3))  # (pr|qs)

    # Spin expansion (BLOCK)
    h1_spin = spin_expand_1e(h1_spatial)               # (2n,2n)
    g2_spin = spin_expand_2e_phys(g_spatial_phys)      # (2n,2n,2n,2n)

    # Build full Fock-space Hamiltonian
    H = hamiltonian_matrix(h1_spin, g2_spin, enuc=mol.energy_nuc())
    return H
