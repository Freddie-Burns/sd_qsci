[![docs](https://img.shields.io/badge/docs-online-blue)](https://freddie-burns.github.io/sd-qsci/)

# Single Determinant QSCI

**Investigating the effectiveness of single Slater determinant trial wavefunctions for QSCI methods.**

This project explores how well simple mean-field states—such as unrestricted or restricted Hartree–Fock (UHF/RHF) single determinants—serve as **trial wavefunctions** in **quantum simulation chemistry (QSCI)** workflows.
The goal is to benchmark their performance and limitations when used in hybrid classical–quantum methods and related post-HF quantum algorithms.

---

## Motivation

Many quantum chemistry and quantum simulation algorithms rely on a trial wavefunction to initialize or constrain a variational search.
While multi-determinant or correlated references can improve accuracy, **single Slater determinants** (from UHF or RHF) remain the simplest and most computationally efficient choice.

This repository aims to:

* Quantify the accuracy of single-determinant trials across small molecules.
* Compare UHF vs RHF determinants as initial states.
* Interface **PySCF** (for reference and integral generation) with **Qiskit** (for quantum simulation).

---

## Project structure

```
sd-qsci/
├─ pyproject.toml         # Project metadata and dependencies (managed by uv)
├─ src/
│  └─ sd_qsci/            # Python package (imports as sd_qsci)
│     ├─ __init__.py
│     ├─ qc.py            # Quantum circuit creation and execution
│     ├─ spin.py          # Spin analysis utilities
│     ├─ utils.py         # General utilities
│     └─ hamiltonian/     # Hamiltonian construction from PySCF
├─ research/              # Research experiments and workflows
│  ├── __init__.py
│  ├── experiments/       # Standalone research scripts
│  └── config/            # Shared configurations
├─ tests/                 # Unit and integration tests
├─ notebooks/
│  └─ dev/                # Development notebooks
├─ docs/                  # Sphinx documentation
├─ data/                  # Molecular geometries and results
└─ README.md
```

---

## Installation

This project uses [**uv**](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/Freddie-Burns/sd-qsci.git
cd sd-qsci

# Install all dependencies (creates virtual environment automatically)
uv sync

# Install with dev dependencies (includes Sphinx, JupyterLab, etc.)
uv sync --all-groups
```

### Main dependencies

* [PySCF](https://pyscf.org) — ab initio electronic structure calculations
* [Qiskit](https://qiskit.org) — quantum circuit construction and simulation
* [ffsim](https://github.com/qiskit-community/ffsim) — fermion simulation for quantum circuits
* [NumPy](https://numpy.org) — numerical computing
* [SciPy](https://scipy.org) — scientific computing
* [pytest](https://docs.pytest.org) — testing framework

---

## Quick start

Launch Jupyter notebooks to explore examples:

```bash
uv run jupyter lab
```

Example notebooks:

* `notebooks/dev/00_hamiltonian_tutorial.ipynb` — Hamiltonian construction from PySCF molecular systems
* `notebooks/dev/01_verify_unitary_qc.ipynb` — Verifies unitarity of quantum circuits and orbital rotation operations
* `notebooks/dev/02_verify_unitary_qc.ipynb` — Uses `qc.py` functions for circuit creation and simulation

---

## Testing

Run tests via:

```bash
uv run pytest
```

---

## Documentation

API documentation is built with [Sphinx](https://www.sphinx-doc.org/) and hosted at:  
**https://freddie-burns.github.io/sd-qsci/**

### Building docs locally

```bash
# Install dev dependencies (if not already done)
uv sync --all-groups

# Build HTML documentation
uv run python -m sphinx -b html docs/source docs/_build/html

# View the built docs
# On Linux/WSL:
xdg-open docs/_build/html/index.html
# On macOS:
open docs/_build/html/index.html
# On Windows:
start docs/_build/html/index.html
```

The documentation includes:
- Full API reference with auto-generated function signatures
- Module documentation for `hamiltonian`, `qc`, `spin`, and `utils`
- Type hints and NumPy-style docstrings

---

## Possible extensions

* Benchmark larger systems (LiH, BeH₂, H₂O).
* Compare RHF vs UHF vs multi-determinant expansions.
* Compare with Qiskit Nature Hamiltonian generation.
* Add visualization of orbital correlation and overlap metrics.

---

## References

* *Qiskit Nature Documentation*
* *PySCF: The Python-based Simulations of Chemistry Framework*
* Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory* (2000)

---

## License

MIT License © [Freddie Burns] 2025



