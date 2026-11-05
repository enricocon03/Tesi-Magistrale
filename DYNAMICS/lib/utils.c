#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <lapacke.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>


#include "../include/random.h"
#include "../include/boxmuller.h"
#include "../include/utils.h"  


double gauss_mu_W(double mu, double W)
{
    double x = gauss1();  // N(0,1)
    return mu + W * x;    // N(mu, W^2)
}

double E_beta(const double *w, const double beta, const unsigned int size)
{
    // w ordered ascending: shift to avoid overflow in exp
    double s;
    if (beta >= 0.0) s = w[0];
    else s = w[size-1];
    
    double Z = 0.0, num = 0.0;
    for (unsigned int n = 0; n < size; n++)
    {
        double p = exp(-beta * (w[n] - s));
        Z   += p;
        num += w[n] * p;
    }
    return num / Z;
}

// Find beta such that E_beta(beta) = E_target
double find_beta(const double *w, const unsigned int size, const double E_target, const double beta_max)
{
    double lo = -beta_max, hi = beta_max;
    for (int it = 0; it < 100; it++)
    {
        double mid = 0.5 * (lo + hi);
        if (E_beta(w, mid, size) > E_target) lo = mid;   
        else hi = mid;
    }
    return 0.5 * (lo + hi);
}

void build_rho_beta(double complex *rho_beta_A, const double beta, const double *O, const double *w, const unsigned int N, const unsigned int NA, const unsigned int NB)
{
    unsigned int n, b, i, j;
    double *pw = malloc(N * sizeof(double));
    if (!pw) { perror("malloc pw"); exit(EXIT_FAILURE); }

    double wmin;
    if (beta>= 0.0) wmin = w[0];
    else wmin = w[N-1];

    double Z = 0.0;
    for (n = 0; n < N; n++)
    {
        pw[n] = exp(-beta * (w[n] - wmin));
        Z += pw[n];
    }
    for (n = 0; n < N; n++) pw[n] /= Z;

    for(b=0; b<NB; b++)
        for(n=0; n<N; n++)
            for(i=0; i<NA; i++)
                for(j=0; j<NA; j++)
                    rho_beta_A[IDX(i,j,NA)] += pw[n]* O[IDX(i+b*NA, n,N)] * O[IDX(j+b*NA,n,N)];

    free(pw);
}

void build_rho_beta_sym(double complex *rho_beta_A, const double beta, const unsigned int *basis_sector, const double *O, const double *w, const unsigned int N, const unsigned int NA)
{
    unsigned int n, i;
    double *pw = malloc(N * sizeof(double));
    if (!pw) { perror("malloc pw"); exit(EXIT_FAILURE); }

    double wmin;
    if (beta>= 0.0) wmin = w[0];
    else wmin = w[N-1];

    double Z = 0.0;
    for (n = 0; n < N; n++)
    {
        pw[n] = exp(-beta * (w[n] - wmin));
        Z += pw[n];
    }
    for (n = 0; n < N; n++) pw[n] /= Z;

    for (i = 0; i < N; i++)
    {
        double d = 0.0;
        const double *Hi = &O[(size_t)i*N];            // i-th row
        for (n = 0; n < N; n++) d += pw[n] * Hi[n] * Hi[n];
        unsigned int s = basis_sector[i] & (NA - 1u);  // A configuration
        rho_beta_A[IDX(s, s, NA)] += d;
    }

    free(pw);
}

void U_v(double complex *v_t, const double *O, const double *M, const double complex *phase, const unsigned int size)
{
    unsigned int i, k;
    double complex sum;
    for (i=0; i<size; i++)
    {
        sum = 0.0 + 0.0*I;
        
        for(k=0; k<size; k++) sum += O[IDX(i,k,size)] * phase[k] * M[k];
        
        v_t[i] = sum;
    }
}

void reduced_density_matrix(double complex *rho_A, const double complex *psi, const unsigned int NA, const unsigned int NB)
{
    unsigned int nB, i, j;
    memset(rho_A, 0, NA * NA * sizeof(double complex));

    for (i = 0; i < NA; i++)
        for (j = 0; j < NA; j++)
            for (nB = 0; nB < NB; nB++)
                rho_A[IDX(i,j,NA)] += psi[i + nB*NA] * conj(psi[j + nB*NA]);
}

double von_neumann_entropy(const double complex *psi, const unsigned int L)
{
    int info;
    unsigned int NA = 1 << (L/2); // dim subspace A
    unsigned int NB = 1 << (L/2); // dim subspace B
    
    double complex *rho_A = malloc(NA * NA * sizeof(double complex));
    if (!rho_A) { perror("malloc rho_A"); exit(EXIT_FAILURE); }

    double *lambda = malloc(NA * sizeof(double));
    if (!lambda) { perror("malloc lambda"); exit(EXIT_FAILURE); }

    reduced_density_matrix(rho_A, psi, NA, NB);
    
    info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'N', 'U', NA, rho_A, NA, lambda);
    if (info != 0) { printf("Error in zheevd: info = %d\n", info); exit(EXIT_FAILURE); }

    double S = 0.0;
    for (unsigned int k = 0; k < NA; k++)
    {
        if (lambda[k] > 1e-14)   // to avoid 0*log(0)
            S -= lambda[k] * log(lambda[k]);
    }

    free(rho_A);
    free(lambda);
    return S;
}

/* Reduced density matrix rho_A = Tr_B |psi><psi|, for a state psi defined ONLY on
   the S^z_tot = 0 sector (length Dsec = N, ordered as basis_sector[]).
 
   Conventions:
   - L  = total number of sites, LA = number of sites in subsystem A
   - NA = 2^LA, NB = 2^(L-LA) are the dimensions of spaces of A and B
   - a basis state s is split as s = a + b*NA, where
     a = s & maskA  (low LA bits  -> sites 0..LA-1, subsystem A)
     b = s >> LA    (high bits    -> sites LA..L-1, subsystem B)
 
   With |psi> = sum_k psi[k] |a_k>_A |b_k>_B, the partial trace over B gives
   rho_A[a,a'] = sum_b sum_{k,l : b_k = b_l = b, a_k = a, a_l = a'} psi[k] * conj(psi[l])
   i.e. only pairs of sector states sharing the same B configuration contribute.
 
   Since states with the same b have the same number of up spins in A (the total is
   fixed), the sum over B automatically makes rho_A block-diagonal in n_A: the entries
   outside the blocks stay zero without any special treatment.
*/
void reduced_density_matrix_sector(double complex *rho_A, const double complex *psi,
                                    const unsigned int *basis_sector, const unsigned int Dsec,
                                    const unsigned int L, const unsigned int LA)
{
    unsigned int NA = 1u << LA;       // dimension of the Hilbert space of A
    unsigned int NB = 1u << (L - LA); // dimension of the Hilbert space of B
    unsigned int maskA = NA - 1u;     // LA low bits set to 1: extracts the A part of a configuration
    memset(rho_A, 0, NA*NA*sizeof(double complex));

    /* Group the sector indices k by the value b of the B configuration, using one
       linked list per b, stored in two plain arrays (no per-node allocation):
       head[b] = index k of the first element of the list for b (-1 if empty)
       next[k] = index of the element following k in its list (-1 = end of list)
       aval[k] = A configuration of the k-th sector state
    */
    int *head = malloc(NB * sizeof(int));
    int *next = malloc(Dsec * sizeof(int));
    unsigned int *aval = malloc(Dsec * sizeof(unsigned int));
    for (unsigned int x = 0; x < NB; x++) head[x] = -1; // all lists start empty

    for (unsigned int k = 0; k < Dsec; k++) 
    {
        unsigned int s = basis_sector[k];  // full configuration of the k-th sector state
        aval[k] = s & maskA;               // A part (low LA bits)
        unsigned int b = s >> LA;          // B part (high bits)
        next[k] = head[b];                 // k points to the previous head of list b...
        head[b] = (int) k;                 // ...and becomes the new head
    }

    /* Partial trace: for each B configuration b, loop over all pairs (k,l) of sector
       states having that b, and accumulate psi[k]*conj(psi[l]) in rho_A[a_k, a_l].
       Pairs with different b never contribute (<b|b'> = delta_bb'). 
    */
    for (unsigned int b = 0; b < NB; b++)
        for (int k = head[b]; k != -1; k = next[k])       // walk the list of b
            for (int l = head[b]; l != -1; l = next[l])   // second walk over the same list
                rho_A[IDX(aval[k], aval[l], NA)] += psi[k] * conj(psi[l]);

    free(head); free(next); free(aval);
}

double von_neumann_entropy_sector(const double complex *psi, const unsigned int *basis_sector, const unsigned int Dsec, const unsigned int L)
{
    const unsigned int LA = L/2;
    int info;
    unsigned int NA = 1u << LA;

    double complex *rho_A = malloc(NA * NA * sizeof(double complex));
    if (!rho_A) { perror("malloc rho_A"); exit(EXIT_FAILURE); }
    double *lambda = malloc(NA * sizeof(double));
    if (!lambda) { perror("malloc lambda"); exit(EXIT_FAILURE); }

    reduced_density_matrix_sector(rho_A, psi, basis_sector, Dsec, L, LA);

    info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'N', 'U', NA, rho_A, NA, lambda);
    if (info != 0) { printf("Error in zheevd: info = %d\n", info); exit(EXIT_FAILURE); }

    double S = 0.0;
    for (unsigned int k = 0; k < NA; k++)
        if (lambda[k] > 1e-14) S -= lambda[k] * log(lambda[k]);

    free(rho_A);
    free(lambda);
    return S;
}

double Uhlmann_fidelity(const double complex *rho_beta, const double complex *rho_t, const unsigned int NA)
{
    // Remember: density matrices are hermitian and positive semidefinite (eigenvalues are probabilities). 
    // Hence any negative eigenvalue of rho_beta or of M is just numerical noise and is set to 0.
    int info;
    unsigned int i, j, k;

    double *lambda = malloc(NA * sizeof(double));
    // Copy of rho_beta (zheevd overwrites it)
    double complex *rho_beta_copy = malloc(NA * NA * sizeof(double complex));
    for (i=0; i<NA; i++)
        for (j=0; j<NA; j++) rho_beta_copy[IDX(i,j,NA)] = rho_beta[IDX(i,j,NA)];
    
    // Diagonalization of rho_beta: rho_beta = U * diag(lambda) * U†
    info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', NA, rho_beta_copy, NA, lambda);
    if (info != 0) { fprintf(stderr, "Error in zheevd (rho_beta): info = %d\n", info); exit(EXIT_FAILURE); }
    // Now rho_beta_copy has the eigenvectors as columns
    // So U[i,k] = rho_beta_copy[IDX(i,k,NA)]

    // Let's build sqrt(rho_beta) = U * diag(sqrt(lambda)) * U† ---
    double complex *sqrt_rho = calloc(NA * NA, sizeof(double complex));

    for (k = 0; k < NA; k++)
    {
        double sq = 0.0;
        if (lambda[k] > 0.0) sq = sqrt(lambda[k]);

        for (i = 0; i < NA; i++)
            for (j = 0; j < NA; j++)
                sqrt_rho[IDX(i,j,NA)] += sq * rho_beta_copy[IDX(i,k,NA)] * conj(rho_beta_copy[IDX(j,k,NA)]);
    }

    // M = sqrt_rho * rho_t * sqrt_rho 
    // Step 1: tmp = rho_t * sqrt_rho
    double complex *tmp = calloc(NA * NA, sizeof(double complex));
    double complex *M   = calloc(NA * NA, sizeof(double complex));

    for (i = 0; i < NA; i++)
        for (j = 0; j < NA; j++)
            for (k = 0; k < NA; k++)
                tmp[IDX(i,j,NA)] += rho_t[IDX(i,k,NA)] * sqrt_rho[IDX(k,j,NA)];

    // Step 2: M = sqrt_rho * tmp
    for (i = 0; i < NA; i++)
        for (j = 0; j < NA; j++)
            for (k = 0; k < NA; k++)
                M[IDX(i,j,NA)] += sqrt_rho[IDX(i,k,NA)] * tmp[IDX(k,j,NA)];

    // Eigenvalues of M (real since M is hermitian)
    double *mu = malloc(NA * sizeof(double));
    info = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'N', 'U', NA, M, NA, mu);
    if (info != 0) { fprintf(stderr, "Error in zheevd (M): info = %d\n", info); exit(EXIT_FAILURE); }

    // G = Tr sqrt(M)
    double trace_sqrt = 0.0;
    for (k = 0; k < NA; k++)
    {
        if (mu[k] > 0.0) trace_sqrt += sqrt(mu[k]);
    }
    double G = trace_sqrt; 

    free(rho_beta_copy); free(lambda);
    free(sqrt_rho); free(tmp); free(M); free(mu);

    return G;
}

void buildH_toy(double *H, const int8_t *basis, const double W, const double xi,
                 const unsigned int L, const unsigned int N)
{
    memset(H, 0, N*sizeof(double));
    unsigned int i, j, n, d;
    double *h_site = malloc(L*sizeof(double));
    double *J = malloc(L*L* sizeof(double));
    memset(J, 0, L*L*sizeof(double));
    
    for (i = 0; i < L; i++) h_site[i] = W*(2.0*myrand() - 1.0);

    for (i = 0; i < L; i++)
        for (j = i+1; j < L; j++)
        {
            d = j - i;
            J[IDX(i,j,L)] = gauss_mu_W(0.0, exp(-(double)d/xi));
        }

    for (n = 0; n < N; n++)
    {
        double diag = 0.0;
        for (i = 0; i < L; i++)
        {
            // on-site potential (along z-axis)
            int8_t s_i = basis[IDX(n,i,L)];
            if(s_i==1) diag -= h_site[i];
            else diag += h_site[i];
            
            // sigmaz-sigmaz
            for (j = i+1; j < L; j++)
            {
                int8_t s_j = basis[IDX(n,j,L)];
                if(s_i==s_j) diag += J[IDX(i,j,L)];
                else diag -= J[IDX(i,j,L)];
            }
        }
        H[n] = diag;
    }
    free(h_site); free(J);
}

void buildH_XXZ_sym(double *H, const int8_t *basis, const double Jx, const double Jz, const double h, const double *u,
                    const int *index_map, const unsigned int L, const unsigned int N)
{
	// Build XXZ Hamiltonian in the S^z_tot = 0 symmetry sector
    memset(H, 0, N*N*sizeof(double));
    int i, n, ipp, exc, mask, jb;
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

double fidelity(const double complex *psi1, const double complex *psi2, const unsigned int size)
{
    unsigned int i;
    double complex sum = 0.0 + 0.0*I;
    double F;
    for(i=0; i<size; i++)
    {
        sum += conj(psi1[i]) * psi2[i];
    }

    F = creal(conj(sum)*sum);
    return F;
}