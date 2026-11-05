#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <lapacke.h>
#include <unistd.h>

#include "../STATICS/include/random.h"
#include "../STATICS/include/utils.h"

#define N 20
#define STRING_LENGTH 70

int main(int argc, char **argv)
{   
    int info;
    double epsilon=1.0, gamma, gaussian, mu=0.0, W=1.0, result;
    unsigned int i, j, a, A, n;
    char datafile[STRING_LENGTH];
    FILE *fp;

    // random number generator seeds
    const unsigned long int seed1=(unsigned long int) time(NULL); 
    const unsigned long int seed2=seed1+127;

    // initialize lib/random.c, use myrand() to generate a random number uniformly distributed in in [0,1) 
    myrand_init(seed1, seed2);
    // use y = gauss_mu_W(mu, W) to generate a random number distributed as a gaussian with mean mu and variance W^2
    
    double *H = malloc(N * N * sizeof(double));
    if (!H) { perror("malloc H"); exit(EXIT_FAILURE); }  
    double *H0 = malloc(N * N * sizeof(double));
    if (!H0) { perror("malloc H0"); exit(EXIT_FAILURE); }
    double *V = malloc(N * N * sizeof(double));
    if (!V) { perror("malloc V"); exit(EXIT_FAILURE); }
    double *w = malloc(N * sizeof(double));
    if (!w) { perror("malloc w"); exit(EXIT_FAILURE); }

    if (argc != 4)
    {
        fprintf(stdout, "How to use this program:\n");
        fprintf(stdout, "  %s gamma A datafile \n\n", argv[0]);
        fprintf(stdout, "  gamma: coupling parameter\n");
        fprintf(stdout, "  A: number of disorder realizations \n");
        fprintf(stdout, "  datafile: name of the file where save r_n \n");
        fprintf(stdout, "  dimension is fixed by macro: N=%d \n", N);
        return EXIT_SUCCESS;
    }
    else
    {
        // read input values
        gamma = atof(argv[1]);
        A = atoi(argv[2]); 

        if(strlen(argv[3]) >= STRING_LENGTH)
		{
			fprintf(stderr, "File name too long. Increase STRING_LENGTH or shorten the name (%s, %d)\n", __FILE__, __LINE__);
		    return EXIT_FAILURE;
		} 
		else strcpy(datafile, argv[3]);
    }

    double r=0., r_square=0., r_err;
    const unsigned int size_r_n_a = (2*N/3 - N/3);

    fp = fopen(datafile, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file %s\n", datafile);
        return EXIT_FAILURE;
    }
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
                    gaussian = gauss_mu_W(mu, sqrt(2.0)*W); // double variance on the diagonal
                else
                    gaussian = gauss_mu_W(mu, W);
                
                V[IDX(i,j,N)] = gaussian;
                V[IDX(j,i,N)] = gaussian;
            }
        }

        for(i=0; i<N; i++)
            for(j=0; j<N; j++)
                H[IDX(i,j,N)] = H0[IDX(i,j,N)] + epsilon * pow(N, -0.5*gamma)*V[IDX(i,j,N)];
            
        info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', N, &H[IDX(0,0,N)], N, w);
        if (info != 0) { fprintf(stdout, "Error in H diagonalization: info = %d\n", info); return EXIT_FAILURE; }
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
    fprintf(stdout, " r = %.6lf +/- %.6lf \n", r, r_err);

    return EXIT_SUCCESS;
}