#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <lapacke.h>
#include <stdint.h>
#include <unistd.h>

#include "../STATICS/include/random.h"
#include "../STATICS/include/utils.h"

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
    // initialize lib/random.c, use myrand() to generate a random number uniformly distributed in in [0,1) 
    myrand_init(seed1, seed2);

    unsigned int a, A, i, itemp, j;
    int info, n;
    double Jx=1.0, Jz, h, result;
    char datafile[STRING_LENGTH];
    FILE *fp;

    if (argc != 5)
    {
        fprintf(stdout, "How to use this program:\n");
        fprintf(stdout, "  %s Jz h A datafile \n\n", argv[0]);
        fprintf(stdout, "  Jz: coupling along z \n");
        fprintf(stdout, "  h: strength of the random field (along z) \n");
        fprintf(stdout, "  A: number of disorder realizations \n");
        fprintf(stdout, "  datafile: name of the file where save r_n \n");
        fprintf(stdout, "  LENGTH is fixed by macro: now it is  %d \n", L);
        fprintf(stdout, " we use PBC \n");
        return EXIT_SUCCESS;
    }
    else
    {
        // read input values
        Jz = atof(argv[1]);
        h = atof(argv[2]);
        A = atoi(argv[3]); 
        
        if(strlen(argv[4]) >= STRING_LENGTH)
		{
			fprintf(stderr, "File name too long. Increase STRING_LENGTH or shorten the name (%s, %d)\n", __FILE__, __LINE__);
		    return EXIT_FAILURE;
		} 
		else strcpy(datafile, argv[4]); 
    }

    int8_t *basis = malloc(full_dim(L)*L * sizeof(int8_t));
    if (!basis) { perror("malloc basis"); exit(EXIT_FAILURE); }
    int *index_map = malloc(full_dim(L) * sizeof(int));
    if (!index_map) { perror("malloc index_map"); exit(EXIT_FAILURE); }
    memset(index_map,-1,full_dim(L) *sizeof(int));

    int count = 0;
    // Build computational basis
    for (i = 0; i < full_dim(L); i++) 
    {
        itemp = i;
        for (j = 0; j < L; j++)
        {
            basis[IDX(i,j,L)] = itemp % 2;        
            itemp = itemp / 2;
        }

        if (__builtin_popcount(i) == L/2) index_map[i] = count++;
        
    }
    const unsigned int N = count;

    double *H = malloc(N * N * sizeof(double));
    if (!H) { perror("malloc H"); exit(EXIT_FAILURE); }
    double *w = malloc(N * sizeof(double));
    if (!w) { perror("malloc w"); exit(EXIT_FAILURE); }
    double *u = malloc(L * sizeof(double));
    double r=0., r_square=0., r_err;
    const unsigned int size_r_n_a = (2*N/3 - N/3);
   
    fp = fopen(datafile, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", datafile);
        return EXIT_FAILURE;
    } 
    for (a=0; a<A; a++)
    {
        for (i = 0; i < L; i++) u[i] = 2.0*myrand() - 1.0; //U[-1,1)
        buildH_XXZ_sym(H, basis, Jx, Jz, h, u, index_map, L, N);
        
        info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', N, &H[IDX(0,0,N)], N, w);
        if (info != 0) { printf("Error in H diagonalization: info = %d\n", info); return EXIT_FAILURE; }
        result = r_a(w, N);
        r += result / (double) A;
        r_square += result*result / (double) A;

        double *r_n_a = malloc(size_r_n_a * sizeof(double));
        if (!r_n_a) { perror("malloc r_n_a"); exit(EXIT_FAILURE); }
        buildP_r(r_n_a, w, N);
        
        for(n=0; n<size_r_n_a; n++) fprintf(fp, "%.4lf \n", r_n_a[n]);

        free(r_n_a);
    }
    fclose(fp);
    r_err = (r_square - r*r) / (double) (A-1);
    r_err = sqrt(r_err);
    fprintf(stdout, "r = %.6lf +/- %.6lf \n", r, r_err);
    
    return EXIT_SUCCESS;
}
