#ifndef RANDOM_H
#define RANDOM_H

#ifdef __cplusplus
extern "C" {
#endif



// initialize the random number generator
void myrand_init(unsigned long int initstate, unsigned long int initseq);

// return a random number in [0,1)
double myrand(void);
#ifdef __cplusplus
}
#endif

#endif
