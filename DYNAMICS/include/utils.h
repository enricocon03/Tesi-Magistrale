#ifndef UTILS_H
#define UTILS_H

#include <complex.h>
#include <stdint.h>


#define IDX(i,j,size) ((i) * (size) + (j))
#define full_dim(L) (1UL << (L))


double gauss_mu_W(double mu, double W);

double E_beta(const double *w, const double beta, const unsigned int size);

double find_beta(const double *w, const unsigned int size, const double E_target, const double beta_max);

void build_rho_beta(double complex *rho_beta_A, const double beta, const double *O, const double *w, const unsigned int N, const unsigned int NA, const unsigned int NB);

void build_rho_beta_sym(double complex *rho_beta_A, const double beta, const unsigned int *basis_sector, const double *O, const double *w, const unsigned int N, const unsigned int NA);

void U_v(double complex *v_t, const double *O, const double *M, const double complex *phase, const unsigned int size);

void reduced_density_matrix(double complex *rho_A, const double complex *psi, const unsigned int NA, const unsigned int NB);

double von_neumann_entropy(const double complex *psi, const unsigned int L);

void reduced_density_matrix_sector(double complex *rho_A, const double complex *psi,
                                    const unsigned int *basis_sector, const unsigned int Dsec,
                                    const unsigned int L, const unsigned int LA);

double von_neumann_entropy_sector(const double complex *psi, const unsigned int *basis_sector, const unsigned int Dsec, const unsigned int L);

double Uhlmann_fidelity(const double complex *rho_beta, const double complex *rho_t, const unsigned int NA);
                    
void buildH_toy(double *H, const int8_t *basis, const double W, const double xi,
                 const unsigned int L, const unsigned int N);

void buildH_XXZ_sym(double *H, const int8_t *basis, const double Jx, const double Jz, const double h, const double *u,
                    const int *index_map, const unsigned int L, const unsigned int N);

double r_a(double *w, const unsigned int size);

double fidelity(const double complex *psi1, const double complex *psi2, const unsigned int size);

#endif
