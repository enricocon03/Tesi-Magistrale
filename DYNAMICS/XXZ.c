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

    unsigned int a, t, A, Nt, i, itemp, k, j, index;
    int info;
    double Jx=1.0, Jz, h, result, t_min, t_max;
    char datafile_S[STRING_LENGTH], datafile_F[STRING_LENGTH];
    FILE *fp_S, *fp_F;

    if (argc != 9)
    {
        fprintf(stdout, "How to use this program:\n");
        fprintf(stdout, "  %s Nt t_min t_max Jz h A datafile_S datafile_F \n\n", argv[0]);
        fprintf(stdout, "  Nt: number of temporal steps \n");
        fprintf(stdout, "  t_min, t_max: extremes of the time line \n");
        fprintf(stdout, "  Jz: coupling along z \n");
        fprintf(stdout, "  h: strength of the random field (along z) \n");
        fprintf(stdout, "  A: number of disorder realizations \n");
        fprintf(stdout, "  datafile_S: name of the file where save S_A \n");
        fprintf(stdout, "  datafile_F: name of the file where save fidelity \n");
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
		else strcpy(datafile_S, argv[7]);

        if(strlen(argv[8]) >= STRING_LENGTH)
		{
			fprintf(stderr, "File name too long. Increase STRING_LENGTH or shorten the name (%s, %d)\n", __FILE__, __LINE__);
		    return EXIT_FAILURE;
		}
		else strcpy(datafile_F, argv[8]);
    }
    
    int8_t *basis = malloc(full_dim(L)*L * sizeof(int8_t));
    if (!basis) { perror("malloc basis"); exit(EXIT_FAILURE); }
    int *index_map = malloc(full_dim(L) * sizeof(int));
    if (!index_map) { perror("malloc index_map"); exit(EXIT_FAILURE); }
    memset(index_map,-1,full_dim(L) *sizeof(int));
    unsigned int *basis_sector = malloc(full_dim(L) * sizeof(unsigned int));
    if (!basis_sector) { perror("malloc basis_sector"); exit(EXIT_FAILURE); }
    memset(basis_sector, 0, full_dim(L) *sizeof(unsigned int)); 

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
    double *S = malloc(Nt * sizeof(double));
    if (!S) {perror("malloc S"); exit(EXIT_FAILURE);}
    double *S_square = malloc(Nt * sizeof(double));
    if (!S_square) {perror("malloc S_square"); exit(EXIT_FAILURE);}
    double *S_err = malloc(Nt * sizeof(double));
    if (!S_err) {perror("malloc S_err"); exit(EXIT_FAILURE);}
    memset(S, 0, Nt*sizeof(double));
    memset(S_square, 0, Nt*sizeof(double));
    double *F = malloc(Nt * sizeof(double));
    if (!F) {perror("malloc S"); exit(EXIT_FAILURE);}
    double *F_square = malloc(Nt * sizeof(double));
    if (!F_square) {perror("malloc F_square"); exit(EXIT_FAILURE);}
    double *F_err = malloc(Nt * sizeof(double));
    if (!F_err) {perror("malloc F_err"); exit(EXIT_FAILURE);}
    memset(F, 0, Nt*sizeof(double));
    memset(F_square, 0, Nt*sizeof(double));
    double *u = malloc(L * sizeof(double));
    if (!u) { perror("malloc u"); exit(EXIT_FAILURE); }
    double *Hdiag = malloc(N * sizeof(double));
    if (!Hdiag) { perror("malloc Hdiag"); exit(EXIT_FAILURE); }
    double *times = malloc(Nt * sizeof(double));
    if (!times) {perror("malloc times"); exit(EXIT_FAILURE);}
    for (t = 0; t < Nt; t++)
    {
        if (t == 0) times[t] = 0.0;
        else times[t] = t_min * pow(t_max / t_min, (double)(t - 1) / (double)(Nt - 2));
    } 
    double r=0., r_square=0., r_err;
    
    for (a=0; a<A; a++)
    {
        for (i = 0; i < L; i++) u[i] = 2.0*myrand() - 1.0; //U[-1,1)
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
        r += result / (double) A;
        r_square += result*result / (double) A;
        
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
            result = von_neumann_entropy_sector(psi, basis_sector, N, L);
            S[t] += result;
            S_square[t] += result*result;

            result = creal(conj(psi[index]) * psi[index]);
            F[t] += result;
            F_square[t] += result*result;
        }
    }

    r_err = (r_square - r*r) / (double) (A-1);
    r_err = sqrt(r_err);
    fprintf(stdout, " r = %.6lf +/- %.6lf \n", r, r_err);

    fp_S = fopen(datafile_S, "w");
    if (fp_S == NULL) { fprintf(stderr, "Error opening file %s\n", datafile_S); return EXIT_FAILURE; }
    fp_F = fopen(datafile_F, "w");
    if (fp_F == NULL) { fprintf(stderr, "Error opening file %s\n", datafile_F); return EXIT_FAILURE; }    
    for(t=0; t<Nt; t++)
    {
        if(A==1)
        {
            S_err[t] = 0.0;
            F_err[t] = 0.0;
        } 
        else
        {
            S[t] /= (double) A; S_square[t] /= (double) A; 
            S_err[t] = (S_square[t] - S[t]*S[t])/(double) (A-1);
            S_err[t] = sqrt(S_err[t]);
            
            F[t] /= (double) A; F_square[t] /= (double) A; 
            F_err[t] = (F_square[t] - F[t]*F[t])/(double) (A-1);
            F_err[t] = sqrt(F_err[t]);
        }
        fprintf(fp_S, "%.8lf %.12lf %.12lf \n", times[t], S[t], S_err[t]);
        fprintf(fp_F, "%.8lf %.12lf %.12lf \n", times[t], F[t], F_err[t]);
    }
    fclose(fp_S);
    fclose(fp_F);


    return EXIT_SUCCESS;
}
