#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>


#define IDX(i,j,size) ((i) * (size) + (j))
#define full_dim(L) (1UL << (L))

double gauss_mu_W(double mu, double W);
                    
void buildH_XXZ_sym(double *H, const int8_t *basis, const double Jx, const double Jz, const double h, const double *u,
                    const int *index_map, const unsigned int L, const unsigned int N);

void buildP_r(double *r_n, double *w, const unsigned int size);

double r_a(double *w, const unsigned int size);

#endif
