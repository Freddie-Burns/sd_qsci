[![docs](https://img.shields.io/badge/docs-online-blue)](https://freddie-burns.github.io/sd_qsci/)

# Single Determinant QSCI

**Investigating the effectiveness of single Slater determinant trial wavefunctions for quantum simulation and QSCI methods.**

This project explores how well simple mean-field states—such as unrestricted or restricted Hartree–Fock (UHF/RHF) single determinants—serve as **trial wavefunctions** in **quantum simulation chemistry (QSCI)** workflows.
The goal is to benchmark their performance and limitations when used in hybrid classical–quantum methods like VQE and related post-HF quantum algorithms.

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
sd_qsci/
├─ pyproject.toml         # Project metadata and dependencies (managed by uv)
├─ src/
│  └─ sd_qsci/
│     ├─ __init__.py
│     ├─ main.py
│     ├─ utils.py
│     ├─ hamiltonian/     # Generate occupation number vector Hamiltonian operator
│     └─ qsci/            # Run quantum simulations and QSCI
├─ tests/                 # Unit and integration tests
├─ notebooks/             # Interactive walkthroughs and analyses
├─ data/                  # Molecular geometries, basis sets, and results
└─ README.md
```

---

## Environment setup

This project uses [**uv**](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone repository
git clone https://github.com/<yourname>/single_determinant_qsci.git
cd single_determinant_qsci

# Create virtual environment
uv venv

# Install dependencies
uv sync --dev
```

Main dependencies:

* [PySCF](https://pyscf.org) — reference mean-field and integrals
* [Qiskit](https://qiskit.org) — quantum simulation and VQE
* [NumPy](https://numpy.org)
* [pytest](https://docs.pytest.org) — testing framework
* [JupyterLab](https://jupyter.org) — interactive analysis

---

## Quick start

Run the demo script:

```bash
uv run python -m single_determinant_qsci.main
```

or launch notebooks:

```bash
uv run jupyter lab
```

Example notebook:

```
notebooks/01_h2_uhf_trial.ipynb
```

Demonstrates generating an unrestricted Hartree–Fock determinant for H₂ and evaluating its effectiveness in a small VQE circuit.

---

## Testing

Run tests via:

```bash
uv run pytest
```

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

