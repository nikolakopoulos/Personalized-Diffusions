#ifndef PERDIF_FIT_H_
#define PERDIF_FIT_H_

#include "rec_defs.h"

void k_simplex(double *, d_mat_t, double *, int, double, double *, bool);
void single_best(double *, d_mat_t, double *, int, double, double *, bool);
void ppr_grid(double *, d_mat_t, double *, int, double, double *, bool);
void hk_grid(double *, d_mat_t, double *, int, double, double *, bool);
void dictionary(double *, d_mat_t, double *, int, double, double *, bool);
void dictionary_single(double *, d_mat_t, double *, int, double, double *,
                       bool);

double *ppr_coefficients(int);
double *hk_coefficients(int);
double *dict_coefficients(int);

#endif
