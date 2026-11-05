#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <lapacke.h>
#include <unistd.h>
#include <complex.h>

#include "../DYNAMICS/include/random.h"
#include "../DYNAMICS/include/utils.h"

#define L 10
#if (L % 2 != 0)
    #error "L must be even!"
#endif

#define STRING_LENGTH 70

int main(int argc, char **argv) 
{
    // random number generator seeds
    const unsigned long int seed1=(unsigned long int) time(NULL); 
    const unsigned long int seed2=seed1+127;

    // initialize lib/random.c, use myrand() to generate a random number uniformly distributed in [0,1) 
    myrand_init(seed1, seed2);

    unsigned int a, t, A, Nt, i, k, j, index;
    int info;
    double gaussian, mu=0.0, W=1.0, result, t_min, t_max;
    unsigned int n_sub = 1; // sites of the subsystem A
    unsigned int NA = 1 << n_sub; // 2^n_sub
    unsigned int NB = 1 << (L - n_sub); // 2^(L-n_sub)

    char datafile_G[STRING_LENGTH];
    FILE *fp;

    if (argc != 6)
    {
        fprintf(stdout, "How to use this program:\n");
        fprintf(stdout, "  %s Nt t_min t_max A datafile_G \n\n", argv[0]);
        fprintf(stdout, "  Nt: number of temporal steps \n");
        fprintf(stdout, "  t_min, t_max: extremes of the time line \n");
        fprintf(stdout, "  A: number of disorder realizations \n");
        fprintf(stdout, "  datafile_G: name of the file where save fidelity \n");
        fprintf(stdout, "  LENGTH is fixed by macro: now it is  %d \n", L);
        return EXIT_SUCCESS;
    }
    else
    {
        // read input values
        Nt = atoi(argv[1]);
        if (Nt < 3)
        {
            fprintf(stderr, "Nt must be >= 3 \n");
            return EXIT_FAILURE;
        }
        t_min = atof(argv[2]);
        t_max = atof(argv[3]);
        if (t_min <= 0.0 || t_max <= t_min)
        {
            fprintf(stderr, "We need 0 < t_min < t_max \n");
            return EXIT_FAILURE;
        }
        A = atoi(argv[4]);
                      
        if(strlen(argv[5]) >= STRING_LENGTH)
		{
			fprintf(stderr, "File name too long. Increase STRING_LENGTH or shorten the name (%s, %d)\n", __FILE__, __LINE__);
		    return EXIT_FAILURE;
		}
		else strcpy(datafile_G, argv[5]);
    }
    
    const unsigned int N = full_dim(L);
    double *H = malloc(N * N * sizeof(double));
    if (!H) { perror("malloc H"); exit(EXIT_FAILURE); }
    double *w = malloc(N * sizeof(double));
    if (!w) { perror("malloc w"); exit(EXIT_FAILURE); }
    double complex *psi = malloc(N * sizeof(double complex));
    if (!psi) { perror("malloc psi"); exit(EXIT_FAILURE); }
    double *M = malloc(N * sizeof(double));
    if (!M) { perror("malloc M"); exit(EXIT_FAILURE); }
    double complex *phase = malloc(N * sizeof(double complex));
    if (!phase) { perror("malloc phase"); exit(EXIT_FAILURE); }
    double *Hdiag = malloc(N * sizeof(double));
    if (!Hdiag) { perror("malloc Hdiag"); exit(EXIT_FAILURE); }
    double *G = malloc(Nt * sizeof(double));
    if (!G) {perror("malloc S"); exit(EXIT_FAILURE);}
    double *G_square = malloc(Nt * sizeof(double));
    if (!G_square) {perror("malloc G_square"); exit(EXIT_FAILURE);}
    double *G_err = malloc(Nt * sizeof(double));
    if (!G_err) {perror("malloc G_err"); exit(EXIT_FAILURE);}
    memset(G, 0, Nt*sizeof(double));
    memset(G_square, 0, Nt*sizeof(double));
    double *beta_star = malloc(A * sizeof(double));
    if (!beta_star) { perror("malloc beta_star"); exit(EXIT_FAILURE); }

    double *times = malloc(Nt * sizeof(double));
    if (!times) {perror("malloc times"); exit(EXIT_FAILURE);}
    for (t = 0; t < Nt; t++)
    {
        if (t == 0) times[t] = 0.0;
        else times[t] = t_min * pow(t_max / t_min, (double)(t - 1) / (double)(Nt - 2));
    } 
    double r=0., r_square=0., r_err;
    const double beta_max = 10.0;
    const double tol_E = 1e-6;

    for (a=0; a<A; a++)
    {
        memset(H,0,N*N*sizeof(double));
        for(i=0;i<N;i++) // build GOE matrix
        {               
            for(j=i;j<N;j++)
            {
                if (i==j)
                    gaussian = gauss_mu_W(mu, sqrt(2.0)*W);  //double variance on the diagonal
                else
                    gaussian = gauss_mu_W(mu, W);
                
                H[IDX(i,j,N)] = gaussian;
            }
        }

        double tr = 0.0;
        for (i=0; i<N; i++) tr+= H[IDX(i,i,N)];
        tr /= (double) N;

        for (i=0; i<N; i++) H[IDX(i,i,N)] -= tr;
        
        for (i=0; i<N; i++) Hdiag[i] = H[IDX(i,i,N)];

        info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', N, &H[IDX(0,0,N)], N, w);
        if (info != 0) { printf("H diagonalization failed: info = %d\n", info); return EXIT_FAILURE; }
        result = r_a(w, N);
        r += result / (double) A;
        r_square += result*result / (double) A;
        
        for(i=0; i<N; i++) psi[i] = 0.0;
        index = (int) ((double) N *myrand()); //randomly chosen computational-basis initial state
        
        double E = Hdiag[index];
        beta_star[a] = find_beta(w, N, E, beta_max);
        
        double E_check = E_beta(w, beta_star[a], N);
        if (fabs(E_check - E) > tol_E || fabs(beta_star[a]) > beta_max - 1e-6)
        {
            fprintf(stderr, "ERROR a=%u: find_beta does not converge (beta*=%.6lf, E=%.6lf, E_beta=%.6lf, |diff|=%.3e)\n",
                    a, beta_star[a], E, E_check, fabs(E_check - E));
            return EXIT_FAILURE;
        }

        double complex *rho_beta_A = calloc(NA*NA, sizeof(double complex));
        if (rho_beta_A) { perror("malloc rho_beta_A"); exit(EXIT_FAILURE); }

        build_rho_beta(rho_beta_A, beta_star[a], H, w, N, NA, NB);
    
        double complex *rho_A_t = malloc(NA*NA*sizeof(double complex));
        if (rho_A_t) { perror("malloc rho_A_t"); exit(EXIT_FAILURE); }
        for (t=0; t<Nt; t++)
        {
            if(t==0) 
            {
                psi[index] = 1.0;
                for (k=0; k<N; k++) M[k] = H[IDX(index,k,N)];
            }  
            else
            {
                for (k=0; k<N; k++) phase[k] = cexp(-I*w[k]*times[t]);
                U_v(psi, H, M, phase, N);
            }
            reduced_density_matrix(rho_A_t, psi, NA, NB);
            result = Uhlmann_fidelity(rho_beta_A, rho_A_t, NA);

            G[t] += result;
            G_square[t] += result*result;
        }
        free(rho_beta_A); free(rho_A_t);
    }

    r_err = (r_square - r*r) / (double) (A-1);
    r_err = sqrt(r_err);
    fprintf(stdout, " r = %.6lf +/- %.6lf \n", r, r_err);

    fp = fopen(datafile_G, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", datafile_G);
        return EXIT_FAILURE;
    }    
    for(t=0; t<Nt; t++)
    {
        if(A==1) G_err[t] = 0.0;
        else
        {
            G[t] /= (double) A; G_square[t] /= (double) A; 
            G_err[t] = (G_square[t] - G[t]*G[t])/(double) (A-1);
            G_err[t] = sqrt(G_err[t]);
        }
        fprintf(fp, "%.8lf %.12lf %.12lf \n", times[t], G[t], G_err[t]);
    }
    fclose(fp);

    return EXIT_SUCCESS;
}
