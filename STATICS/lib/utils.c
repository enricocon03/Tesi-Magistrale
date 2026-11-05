#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <lapacke.h>
#include <stdint.h>
#include <string.h>

#include "../include/boxmuller.h"
#include "../include/utils.h"  


double gauss_mu_W(double mu, double W)
{
    double x = gauss1();  // N(0,1)
    return mu + W * x;    // N(mu, W^2)
}

void buildH_XXZ_sym(double *H, const int8_t *basis, const double Jx, const double Jz, const double h, const double *u,
                    const int *index_map, const unsigned int L, const unsigned int N)
{
	// Build XXZ Hamiltonian in the S^z_tot = 0 symmetry sector
    memset(H, 0, N*N*sizeof(double));
    int i, n, ipp, exc, mask, jb, jb2;
    int8_t s_i, s_ipp;
    double *h_site = malloc(L* sizeof(double));
    for(i=0; i<L; i++) h_site[i] = h * u[i];
    
    for(i=0; i<L; i++)
    {
        ipp = (i + 1) % L; //first neighbour with PBC
        mask = (1 << i) | (1 << ipp);

        int idx = 0;    

        for(n=0; n<full_dim(L); n++)
        {
            if (__builtin_popcount(n) == L/2)
            {
                s_i  = basis[IDX(n,i,L)];
                s_ipp = basis[IDX(n,ipp,L)];
                
                // on-site field (along z-axis) 
                if(s_i==1) H[IDX(idx,idx,N)] -= h_site[i];
                else H[IDX(idx,idx,N)] += h_site[i];

                // heisenberg coupling i-ipp
                exc = n ^ mask;
                jb = index_map[exc];

                if(jb>=0) //just a check 
                {
                    //sigmax-sigmax
                    H[IDX(jb,idx,N)] += Jx;

                    //sigmay-sigmay
                    if(s_i==s_ipp) H[IDX(jb,idx,N)] -= Jx;
                    else H[IDX(jb,idx,N)] += Jx;
                }

                // sigmaz-sigmaz
                if(s_i==s_ipp) H[IDX(idx,idx,N)] += Jz;
                else H[IDX(idx,idx,N)] -= Jz;    
                idx++;
            }
        }
    }
    free(h_site);
}

double r_a(double *w, const unsigned int size)
{
    int n, counter;
    double r_a_val, r_a_n, delta_n, delta_npp;
    
    counter = 0;
    r_a_val = 0.;

    for(n=(size/3); n<(2*size/3); n++)
    {
        delta_npp = fabs(w[n+1]- w[n]);
        delta_n = fabs(w[n] - w[n-1]);

        if(delta_n > delta_npp) r_a_n = delta_npp / delta_n;
        else r_a_n = delta_n / delta_npp;
        
        r_a_val += r_a_n;
        counter++;         
    }

    r_a_val /= (double) counter;
    return r_a_val;
}

void buildP_r(double *r_n, double *w, const unsigned int size)
{
    int n;
    double r_a_n, delta_n, delta_npp;
    
    for(n=(size/3); n<(2*size/3); n++)
    {
        delta_npp = fabs(w[n+1]- w[n]);
        delta_n = fabs(w[n] - w[n-1]);

        if(delta_n > delta_npp) r_a_n = delta_npp / delta_n;
        else r_a_n = delta_n / delta_npp;
        
        r_n[n - size/3] = r_a_n;
    }

}
