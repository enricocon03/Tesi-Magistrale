#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <lapacke.h>
#include <stdint.h>
#include <unistd.h>
#include <complex.h>

#include "../DYNAMICS/include/random.h"
#include "../DYNAMICS/include/utils.h"

#define L 10
#if (L % 2 != 0)
    #error "L must be even!"
#endif

#define STRING_LENGTH 70
#define E_TOL 0.1      // tolerance on |E_s - E_c|, in units of the spectrum width
#define MAX_TRIES 1000 // maximum attempts to drawn index

int main(int argc, char **argv) 
{
    // random number generator seeds
    const unsigned long int seed1=(unsigned long int) time(NULL); 
    const unsigned long int seed2=seed1+127;

    // initialize lib/random.c, use myrand() to generate a random number uniformly distributed in [0,1) 
    myrand_init(seed1, seed2);

    unsigned int A, Nt;
    unsigned int n_sub = 1;  // sites of the subsystem A
    unsigned int NA = 1 << n_sub; // 2^n_sub  
    int info, a, t, i, itemp, n, k, j, index;
    double Jx=1.0, Jz, h, result, t_min, t_max, m_beta_a, m_t_a;
    char datafile_G[STRING_LENGTH];
    FILE *fp;

    if (argc != 8)
    {
        fprintf(stdout, "How to use this program:\n");
        fprintf(stdout, "  %s Nt t_min t_max Jz h A datafile_G \n\n", argv[0]);
        fprintf(stdout, "  Nt: number of temporal steps \n");
        fprintf(stdout, "  t_min, t_max: extremes of the time line \n");
        fprintf(stdout, "  Jz: coupling along z \n");
        fprintf(stdout, "  h: strength of the random field (along z) \n");
        fprintf(stdout, "  A: number of disorder realizations \n");
        fprintf(stdout, "  datafile_G: name of the file where save G(rho_A, rho_beta) \n");
        fprintf(stdout, "  LENGTH is fixed by macro: now it is  %d \n", L);
        fprintf(stdout, " we use PBC \n");
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
        Jz = atof(argv[4]);
        h = atof(argv[5]);
        A = atoi(argv[6]);
                      
        if(strlen(argv[7]) >= STRING_LENGTH)
		{
			fprintf(stderr, "File name too long. Increase STRING_LENGTH or shorten the name (%s, %d)\n", __FILE__, __LINE__);
		    return EXIT_FAILURE;
		}
		else strcpy(datafile_G, argv[7]);	
    }

    int8_t *basis = malloc(full_dim(L)*L * sizeof(int8_t));
    if (!basis) { perror("malloc basis"); exit(EXIT_FAILURE); }
    int *index_map = malloc(full_dim(L) * sizeof(int));
    if (!index_map) { perror("malloc index_map"); exit(EXIT_FAILURE); }
    memset(index_map,-1,full_dim(L) *sizeof(int));
    unsigned int *basis_sector = malloc(full_dim(L) * sizeof(unsigned int));
    memset(basis_sector,0,full_dim(L) *sizeof(int));

    int count = 0;
    // Build computational basis
    for (i = 0; i < full_dim(L); i++) 
    {
        itemp = i;
        for (j = 0; j < L; j++) 
        {
            basis[IDX(i,j,L)] = itemp % 2; 
            itemp /= 2; 
        }
        if (__builtin_popcount(i) == L/2) 
        {
            basis_sector[count] = i; 
            index_map[i] = count; 
            count++; 
        }
    }  
    const unsigned int N = count;

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
    double *G = malloc(Nt * sizeof(double));
    if (!G) {perror("malloc G"); exit(EXIT_FAILURE);}
    double *G_square = malloc(Nt * sizeof(double));
    if (!G_square) {perror("malloc G_square"); exit(EXIT_FAILURE);}
    double *G_err = malloc(Nt * sizeof(double));
    if (!G_err) {perror("malloc G_err"); exit(EXIT_FAILURE);}
    memset(G, 0, Nt*sizeof(double));
    memset(G_square, 0, Nt*sizeof(double));

    double *u = malloc(L * sizeof(double));
    if (!u) { perror("malloc u"); exit(EXIT_FAILURE); }

    double *Deltam_t = malloc(Nt * sizeof(double));
    if (!Deltam_t) { perror("malloc Deltam_t"); exit(EXIT_FAILURE); }
    double *Deltam_t_square = malloc(Nt * sizeof(double));
    if (!Deltam_t_square) { perror("malloc Deltam_t_square"); exit(EXIT_FAILURE); }
    double *Deltam_t_err = malloc(Nt * sizeof(double));
    if (!Deltam_t_err) { perror("malloc Deltam_t_err"); exit(EXIT_FAILURE); }
    memset(Deltam_t, 0, Nt*sizeof(double));
    memset(Deltam_t_square, 0, Nt*sizeof(double));

    double *Hdiag = malloc(N * sizeof(double));
    if (!Hdiag) { perror("malloc Hdiag"); exit(EXIT_FAILURE); }
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
        for (i = 0; i < L; i++) u[i] = 2.0*myrand() - 1.0;
        buildH_XXZ_sym(H, basis, Jx, Jz, h, u, index_map, L, N);
        double Ec = 0.0, TrH2 = 0.0;
        for (i = 0; i < N; i++)
        {
            Hdiag[i] = H[IDX(i,i,N)]; // <s|H|s>
            Ec += Hdiag[i];
            TrH2 += Hdiag[i]*Hdiag[i];
            for (j = i+1; j < N; j++) TrH2 += 2.0 * H[IDX(i,j,N)] * H[IDX(i,j,N)];
        }
        Ec /= (double) N;                             // center of the spectrum: Tr(H)/N
        double sigma = sqrt(TrH2/(double) N - Ec*Ec); // variance of the spectrum
        
        info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', N, &H[IDX(0,0,N)], N, w);
        if (info != 0) { printf("H diagonalization failed: info = %d\n", info); return EXIT_FAILURE; }
        result = r_a(w, N);
        r += result;
        r_square += result*result;

        for(i=0; i<N; i++) psi[i] = 0.0;
        // initial state: redrawn until the energy is close to the center of the spectrum
        unsigned int best = 0;
        double best_dist = 1e300;
        for (k = 0; k < MAX_TRIES; k++)
        {
            index = (int) ((double) N * myrand());
            double dist = fabs(Hdiag[index] - Ec);
            if (dist < best_dist) { best_dist = dist; best = index; }
            if (dist < E_TOL * sigma) break;
        }
        index = best; // if none respects the threshold, it uses the closer to Ec
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

        build_rho_beta_sym(rho_beta_A, beta_star[a], basis_sector, H, w, N, NA);     
        

        if (n_sub==1) m_beta_a = creal(rho_beta_A[IDX(0,0,NA)]) - creal(rho_beta_A[IDX(1,1,NA)]);
        
        double complex *rho_A_t = malloc(NA*NA*sizeof(double complex));

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
            
            reduced_density_matrix_sector(rho_A_t, psi, basis_sector, N, L, n_sub);
            result = Uhlmann_fidelity(rho_beta_A, rho_A_t, NA);

            G[t] += result;
            G_square[t] += result*result;

            if (n_sub==1)
            {
                m_t_a = creal(rho_A_t[IDX(0,0,NA)]) - creal(rho_A_t[IDX(1,1,NA)]);
                result = fabs(m_t_a - m_beta_a);

                Deltam_t[t] += result;
                Deltam_t_square[t] += result*result;
            }
        }                        
        free(rho_beta_A); free(rho_A_t);
    }

    r /= (double) A;
    r_square /= (double) A;
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
        if(A==1)   
        {
            G_err[t] = 0.0;
            if (n_sub==1) Deltam_t_err[t] = 0.0;
        }
        else
        {
            G[t] /= (double) A; G_square[t] /= (double) A; 
            G_err[t] = (G_square[t] - G[t]*G[t])/(double) (A-1);
            G_err[t] = sqrt(G_err[t]);

            if (n_sub==1)
            {
                Deltam_t[t] /= (double) A; Deltam_t_square[t] /= (double) A; 
                Deltam_t_err[t] = (Deltam_t_square[t] - Deltam_t[t]*Deltam_t[t])/(double) (A-1);
                Deltam_t_err[t] = sqrt(Deltam_t_err[t]);
            }
        }
        if (n_sub==1) fprintf(fp, "%.8lf %.10lf %.10lf %.10lf %.10lf \n", times[t], G[t], G_err[t], Deltam_t[t], Deltam_t_err[t]);
        else fprintf(fp, "%.8lf %.10lf %.10lf \n", times[t], G[t], G_err[t]);
    }
    fclose(fp);

    return EXIT_SUCCESS;
}
