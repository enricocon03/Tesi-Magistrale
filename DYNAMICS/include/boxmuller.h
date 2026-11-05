#ifndef BOXMULLER_H
#define BOXMULLER_H

// single normal random number 
// polar form
double gauss1(void);

// couple of independent normal random numbers
// polar form
void gauss2(double * __restrict__ ris1, double * __restrict__ ris2);

// couple of independent normal random numbers
// basic form
void gauss2_basic(double * __restrict__ ris1, double * __restrict__ ris2);

#endif
