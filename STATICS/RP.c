#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <lapacke.h>
#include <unistd.h>

#include "../STATICS/include/random.h"
#include "../STATICS/include/utils.h"

static inline double xlogx(double p) { return (p > 0.0) ? p * log(p) : 0.0; }
static inline double safe_sqrt_var(double var) { return sqrt(var > 0.0 ? var : 0.0); }

#define N 20
#define P_MIN_KL 1e-12 

int main(int argc, char **argv)
{   
    int info;
    double epsilon=1.0, gamma, gaussian, mu=0.0, W=1.0;
    unsigned int i, j, a, A, n;

    // random number generator seeds
    const unsigned long int seed1=(unsigned long int) time(NULL); 
    const unsigned long int seed2=seed1+127;

    // initialize lib/random.c, use myrand() to generate a random number uniformly distributed in in [0,1) 
    myrand_init(seed1, seed2);
    // use y = gauss_mu_W(mu, W) to generate a random number distributed as a gaussian with mean mu and variance W^2

    double *H = malloc(N * N * sizeof(double));
    if (!H) { perror("malloc H"); exit(EXIT_FAILURE); }  
    double *H_plus = malloc(N * N * sizeof(double));
    if (!H_plus) { perror("malloc H_plus"); exit(EXIT_FAILURE); }
    double *H_minus = malloc(N * N * sizeof(double));
    if (!H_minus) { perror("malloc H_minus"); exit(EXIT_FAILURE); }
    double *H0 = malloc(N * N * sizeof(double));
    if (!H0) { perror("malloc H0"); exit(EXIT_FAILURE); }
    double *V = malloc(N * N * sizeof(double));
    if (!V) { perror("malloc V"); exit(EXIT_FAILURE); }

    double *w = malloc(N * sizeof(double));
    if (!w) { perror("malloc w"); exit(EXIT_FAILURE); }
    double *w_plus = malloc(N * sizeof(double));
    if (!w_plus) { perror("malloc w_plus"); exit(EXIT_FAILURE); }
    double *w_minus = malloc(N * sizeof(double));
    if (!w_minus) { perror("malloc w_minus"); exit(EXIT_FAILURE); }

    if (argc != 3)
    {
        fprintf(stdout, "How to use this program:\n");
        fprintf(stdout, "  %s gamma A \n\n", argv[0]);
        fprintf(stdout, "  gamma: coupling parameter \n");
        fprintf(stdout, "  A: number of disorder realizations \n");
        fprintf(stdout, "  dimension is fixed by macro: N=%d \n", N);
        return EXIT_SUCCESS;
    }
    else
    {
        // read input values
        gamma = atof(argv[1]);
        A = atoi(argv[2]); 
    }

    double r = 0., r_square=0., r_err;
    double S = 0., S_square = 0., S_err;
    double s1 = 0., s1_square = 0., s1_err; 
    double delta_gamma = 1e-2;
    double KL = 0., KL_square = 0., KL_err;
    double IPR = 0., IPR_square = 0., IPR_err;

    for (a=0; a<A; a++)
    {   
        memset(H,0,N*N*sizeof(double));
        memset(H0,0,N*N*sizeof(double));
        memset(V,0,N*N*sizeof(double));
        for(i=0;i<N;i++)
        {          
            H0[IDX(i,i,N)] = gauss_mu_W(mu, W);
            for(j=i;j<N;j++)
            {
                if (i==j)
                    gaussian = gauss_mu_W(mu, sqrt(2.0)*W); //double variance on the diagonal
                else
                    gaussian = gauss_mu_W(mu, W);
                
                V[IDX(i,j,N)] = gaussian;
                V[IDX(j,i,N)] = gaussian;
            }
        }


        for(i=0; i<N; i++)
        {
            for(j=0; j<N; j++)
            {
                H[IDX(i,j,N)] = H0[IDX(i,j,N)] + epsilon * pow(N, -0.5*gamma)*V[IDX(i,j,N)];
                H_minus[IDX(i,j,N)] = H0[IDX(i,j,N)] + epsilon * pow(N, -0.5*(gamma-delta_gamma))*V[IDX(i,j,N)];
                H_plus[IDX(i,j,N)] = H0[IDX(i,j,N)] + epsilon * pow(N, -0.5*(gamma+delta_gamma))*V[IDX(i,j,N)];
            }
        }

        info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', N, &H[IDX(0,0,N)], N, w);
        if (info != 0) { fprintf(stdout, "Error in H diagonalization: info = %d\n", info); return EXIT_FAILURE; }

        info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', N, &H_plus[IDX(0,0,N)], N, w_plus);
        if (info != 0) { fprintf(stdout, "Error in H_plus diagonalization: info = %d\n", info); return EXIT_FAILURE; }

        info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', N, &H_minus[IDX(0,0,N)], N, w_minus);
        if (info != 0) { fprintf(stdout, "Error in H_minus diagonalization: info = %d\n", info); return EXIT_FAILURE; }

        double r_a=0.0, S_a=0.0, KL_a=0.0, s1_a=0.0, ipr_a=0.0;

        unsigned int n_low  = N / 3;
        unsigned int n_high = (2 * N) / 3;         
        unsigned int n_count = n_high - n_low;  
        
        for(n=n_low; n<n_high; n++)
        {
            double delta_npp = fabs(w[n+1]- w[n]);
            double delta_n = fabs(w[n] - w[n-1]);

            if(delta_n > delta_npp) r_a += delta_npp / delta_n;
            else if(delta_n < delta_npp) r_a += delta_n / delta_npp; //we need to exclude the cases of degenerations

            int m = n+1;
            for (i = 0; i < N; i++)
            {
                double p_n = H[IDX(i,n,N)] * H[IDX(i,n,N)];
                double p_m = H[IDX(i,m,N)] * H[IDX(i,m,N)];

                ipr_a += p_n * p_n; 
                S_a -= xlogx(p_n);

                if (p_n > 0.0) 
                {
                    double p_m_reg;
                    if (p_m > P_MIN_KL)
                        p_m_reg = p_m;
                    else
                        p_m_reg = P_MIN_KL;
        
                    KL_a += p_n * log(p_n / p_m_reg);
                }

                double p_n_plus  = H_plus[IDX(i,n,N)]  * H_plus[IDX(i,n,N)];
                double p_n_minus = H_minus[IDX(i,n,N)] * H_minus[IDX(i,n,N)];
                s1_a += (xlogx(p_n_plus) - xlogx(p_n_minus)) / (2.0*delta_gamma) / log((double) N);
            }
        }
        r_a /= (double) n_count;
        S_a /= (double) n_count;
        s1_a /= (double) n_count;
        KL_a /= (double) n_count;
        ipr_a /= (double) n_count;

        r += r_a / (double)A;
        r_square += r_a*r_a / (double)A; 

        S += S_a / (double)A;
        S_square += S_a*S_a / (double)A; 

        s1 += s1_a / (double)A;
        s1_square += s1_a*s1_a / (double)A; 

        KL += KL_a / (double)A;
        KL_square += KL_a*KL_a / (double)A;

        IPR += ipr_a / (double)A; 
        IPR_square += ipr_a*ipr_a / (double)A; 
    }

    r_err   = safe_sqrt_var((r_square - r*r) / (double)(A-1));
    S_err   = safe_sqrt_var((S_square - S*S) / (double)(A-1));
    s1_err  = safe_sqrt_var((s1_square - s1*s1) / (double)(A-1));
    KL_err  = safe_sqrt_var((KL_square - KL*KL) / (double)(A-1));
    IPR_err = safe_sqrt_var((IPR_square - IPR*IPR) / (double)(A-1));

    fprintf(stdout, " r = %.6lf +/- %.6lf \n", r, r_err);
    fprintf(stdout, " S = %.8lf +/- %.8lf \n", S, S_err);
    fprintf(stdout, " s1 = %.8lf +/- %.8lf \n", s1, s1_err);
    fprintf(stdout, " KL = %.8lf +/- %.8lf \n", KL, KL_err);
    fprintf(stdout, " IPR = %.8lf +/- %.8lf \n", IPR, IPR_err);

    return EXIT_SUCCESS;
}