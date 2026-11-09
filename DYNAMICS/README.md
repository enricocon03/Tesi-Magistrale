# DYNAMICS

Codes for the dynamics of the random-matrix ensembles, of the disordered XXZ chain and of toy-model for MBL, discussed in Chapter 4 of the thesis.

## Files

| File | Model | Output |
|------|-------|--------|
| `GOE.c` | Gaussian Orthogonal Ensemble | Entanglement entropy `S_A(t)`, Fidelity `F(t)` |
| `XXZ.c` | Disordered XXZ chain (zero-magnetization sector) | `S_A(t)`, `F(t)` as a function of the disorder h and/or of J_z |
| `RP.c` | Rosenzweig-Porter ensemble | `S_A(t)`, `F(t)` as a function of \gamma |
| `Toy_pert.c` | Toy model for MBL with a GOE-like perturbation | `S_A(t)`, `F(t)` as a function of \xi and \gamma |
| `XXZ_Uhlmann.c` | Disordered XXZ chain | Uhlmann fidelity (and \Delta m(t) if single-site subsystem) |
| `GOE_Uhlmann.c` | Gaussian Orthogonal Ensemble | Uhlmann fidelity |

## Requirements

- C compiler (e.g. `gcc`)
- LAPACK/BLAS (routine `dsyevd` for the full diagonalization), e.g. `liblapack`, `libopenblas`

## Compilation

```bash
gcc -O3 -o XXZ XXZ.c lib/pcg32min.c lib/random.c lib/boxmuller.c lib/utils.c -llapacke -lopenblas -lm
```
(similar for the other files; adapt the flags to your system).

## Usage

```bash
OPENBLAS_NUM_THREADS=<num_threads> ./XXZ <Nt> <t_min> <t_max> <Jz> <h> <n_realizations> <datafile_S> <datafile_F>
```
`OPENBLAS_NUM_THREADS=<num_threads>` limits the threads used by OpenBLAS

(similar for the other files).

## Notes

- For the XXZ chain the Hilbert-space dimension is `N = binom(L, L/2)` (sector `sigma^z_tot = 0`, periodic boundary conditions, `J_x = 1`).
- Cost of the full diagonalization: O(N^3) in time, O(N^2) in memory.
