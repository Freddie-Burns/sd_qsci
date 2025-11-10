# sd_qsci – Development Guidelines (project-specific)

This note captures practical, project-specific details for advanced contributors. It focuses on the real environment used here (uv + PySCF + Qiskit), test execution patterns, and debugging tips relevant to this repository.


## 1. Build and configuration

The project is managed with uv and targets Python 3.12 (see `pyproject.toml`).

- Create and activate the environment (Linux/WSL/Unix):
  - Create venv: `uv venv`
  - Install deps (including dev): `uv sync --dev`
  - One-off run commands in the venv: `uv run <cmd>`

- Python version: `requires-python = ">=3.12"`. Using 3.12 is recommended to match lockfile and dependencies.

- Key runtime dependencies (abbrev.):
  - PySCF (>= 2.11): HF references, integrals
  - Qiskit, Qiskit Aer: simulation backend (not required for all tests)
  - NumPy, SciPy: numerics
  - pytest: test runner

- OS notes:
  - On Linux and WSL2, PySCF wheels typically include necessary binaries. If you build from source, ensure BLAS/LAPACK presence. If NumPy/SciPy use MKL/OpenBLAS, pin threads to avoid oversubscription (see "Performance knobs").
  - On macOS ARM, prefer official wheels; source builds can be more involved.

- Docs toolchain (optional): `sphinx`, `myst-parser`, `furo` are under the `dev` dependency group.


## 2. Tests – running, adding, and an example

### 2.1 Running the test suite

- Run all tests:
  - `uv run pytest`

- Run selected tests (faster inner loop):
  - By file: `uv run pytest tests/test_hamiltonian.py::test_spin_expand_1e_blocks -q`
  - By expression: `uv run pytest -k "spin_expand_1e or create_annihilate" -q`

- PySCF-backed tests can be relatively slow (SCF per test). For quick checks, target the light-weight tests that exercise pure-Python logic (see example below).

- Parallelization: If you add `pytest-xdist`, you can run with `-n auto`, but be mindful of PySCF and BLAS threads. See "Performance knobs" below to avoid CPU oversubscription.

### 2.2 Adding new tests

- Place unit tests under `tests/` with file names `test_*.py`.
- Prefer deterministic, small test cases for Hamiltonian blocks and fermion ops. For tests that depend on external quantum chemistry backends (PySCF), cache or parametrize minimal systems (e.g., H2, STO-3G) and keep tolerances explicit.
- If you add new spin/fermion utilities under `src/sd_qsci/hamiltonian/`, mirror the existing structure and write small tests that do not require PySCF unless necessary.
- Use `numpy.allclose`/`isclose` with explicit `atol`/`rtol`.

### 2.3 Verified minimal example test (fast)

To demonstrate the workflow with a test that does not require PySCF, we temporarily created `tests/_junie_demo_test.py` with the following content:

```python
from sd_qsci.hamiltonian.fermion_ops import create, annihilate


def test_roundtrip_create_annihilate_same_orbital():
    # |0101> over 4 modes -> bits set at 0 and 2
    state = (1 << 0) | (1 << 2)

    res = annihilate(state, 0)
    assert res is not None
    ph1, s1 = res

    res2 = create(s1, 0)
    assert res2 is not None
    ph2, s2 = res2

    assert s2 == state
    assert ph1 * ph2 in (+1, -1)


def test_noop_conditions():
    empty = 0
    assert annihilate(empty, 1) is None

    filled = (1 << 3)
    assert create(filled, 3) is None
```

We executed just this test with:

```
uv run pytest tests/_junie_demo_test.py -q
```

Result observed locally during documentation authoring:
```
..                                                                       [100%]
2 passed in ~2s
```

Per the task requirement, this file was only for demonstration and has been removed after verification. You can recreate it if needed using the snippet above.


## 3. Additional development information

### 3.1 Module layout relevant to tests

- `src/sd_qsci/hamiltonian/fermion_ops.py`
  - Pure-Python bitstring fermion ops: `create`, `annihilate`. Good for fast unit tests.
- `src/sd_qsci/hamiltonian/pyscf_glue.py`
  - Converts PySCF RHF results into spin-orbital Hamiltonian via AO→MO transforms and builds a Fock-space Hamiltonian using `hamiltonian_matrix`.
  - Exposes `hamiltonian_from_pyscf(mol, rhf)`.
- `src/sd_qsci/hamiltonian/spin_blocks.py`
  - Spin-expansion utilities for 1e/2e integrals (BLOCK spin ordering).
- `src/sd_qsci/hamiltonian/fermion_hamiltonian.py`, `checks.py`
  - Construct sparse Hamiltonian on occupation basis and validate energies from MO integrals.

Note: The current `tests/test_hamiltonian.py` imports `occ_hamiltonian_from_pyscf` but the implementation exports `hamiltonian_from_pyscf`. If you encounter an import error, either adapt the test import or add a thin alias in the module. This doc does not change code; it flags the mismatch for developers to resolve in code review.

### 3.2 Performance knobs (numerics/BLAS/SCF)

- Control threads during tests to avoid oversubscription and improve determinism:
  - `export OMP_NUM_THREADS=1`  (or `set OMP_NUM_THREADS=1` on Windows PowerShell)
  - For MKL-backed NumPy/SciPy: `MKL_NUM_THREADS=1` and `NUMEXPR_NUM_THREADS=1`
- PySCF SCF options for tiny systems (inside tests): use `conv_tol`, `max_cycle`, and minimal basis like `sto-3g` to keep runs short and stable.

### 3.3 Style and conventions

- Follow the existing code style in each module (imports, spacing, naming). The codebase favors explicit numpy operations and clear, small functions.
- Tests: keep names descriptive (what is validated) rather than how.
- Document BLOCK spin ordering conventions in any new routines:
  - Spin blocks: `[α0..α(n-1), β0..β(n-1)]`.
  - Two-electron integrals use physicist’s `(pr|qs)` index order internally after transpose.

### 3.4 Rebuilding docs locally (optional)

- Build docs with:
  - `uv run sphinx-build -b html docs/source docs/_build/html`
- Auto-reload (if you installed `sphinx-autobuild`):
  - `uv run sphinx-autobuild docs/source docs/_build/html`


## 4. Known pitfalls

- PySCF version mismatches can change integral conventions or SCF defaults. If you update `pyscf`, re-validate energies in tests with explicit tolerances.
- Qiskit Aer GPU/AVX feature detection can impact reproducibility across machines; pin versions as needed and avoid optional hardware paths in CI.
- Ensure that the BLOCK spin ordering is consistently used across the pipeline (expansion → Hamiltonian build → tests). Mismatches are a common source of sign/order bugs.


## 5. Quick checklist

- [ ] `uv sync --dev` completes without build errors
- [ ] `uv run pytest -q` runs; for quick iterations, target light tests via `-k`
- [ ] Thread caps set for determinism on shared CI/dev machines
- [ ] If editing PySCF glue, double-check AO→MO transforms, tensor index orders, and nuclear energy addition
