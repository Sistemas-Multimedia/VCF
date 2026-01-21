#define _2DVECTOR_

void ifct(double *f, int length);
void fct(double *f, int length);
#ifndef _2DVECTOR_
void fct2d(double f[], int nrows, int ncols);
void ifct2d(double f[], int nrows, int ncols);
#else
void fct2d(double **f, int nrows, int ncols);
void ifct2d(double **f, int nrows, int ncols);
#endif
