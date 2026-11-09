# STATICS

Codes for the static properties (eigenvalues and eigenvectors) of the random-matrix ensembles and of the disordered XXZ chain, discussed in Chapter 3 of the thesis.

## Files

| File | Model | Output |
|------|-------|--------|
| `GOE.c` | Gaussian Orthogonal Ensemble | adjacent gap ratio `r`, participation entropy `S`, Kullback-Leibler divergence `KL`, inverse participation ratio `IPR` |
| `XXZ.c` | Disordered XXZ chain (zero-magnetization sector) | `r`, `S`, `-ds/dh`, `KL`, `IPR` as a function of the disorder h and/or of J_z |
| `RP.c` | Rosenzweig-Porter ensemble | `r`, `S`, `-ds/d\gamma`, `KL`, `IPR` as a function of \gamma |
| `XXZ_pdf_r.c` | Disordered XXZ chain | probability distribution `p(r)` of the adjacent gap ratio |
| `RP_pdf_r.c` | Rosenzweig-Porter ensemble | probability distribution `p(r)` of the adjacent gap ratio |

All quantities are computed on the middle third of the spectrum of each disorder realization and averaged over the realizations; statistical errors are estimated as described in Chapter 3.

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
OPENBLAS_NUM_THREADS=<num_threads> ./XXZ <Jz> <h> <n_realizations>
```
`OPENBLAS_NUM_THREADS=<num_threads>` limits the threads used by OpenBLAS

(similar for the other files).

## Notes

- For the XXZ chain the Hilbert-space dimension is `N = binom(L, L/2)` (sector `sigma^z_tot = 0`, periodic boundary conditions, `J_x = 1`).
- Cost of the full diagonalization: O(N^3) in time, O(N^2) in memory.
