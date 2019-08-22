///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains the PERDIF program that implements personalized diffusions for recommendations.

 Code by: Dimitris Berberidis and Athanasios N. Nikolakopoulos
 University of Minnesota 2018-2019
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
// #include <cblas.h>
#include <mkl.h>

#include "rec_defs.h"
#include "perdif_fit.h"

#if DEBUG
static double cost_func(double *, double *, double *, sz_med_t);
#endif

static void matrix_matrix_product(double *C, double *A, double *B, int, int, int);
static double *unfold_matrix(d_mat_t);
static void constr_QP_with_PG(double *, double *, double *, sz_med_t);
static void grammian_matrix(double *, double *, int, int);
static double max_abs_dif(double *, double *, sz_long_t);
// static void project_to_simplex(double *, sz_med_t);
// static void project_to_simplex_thanos(double *, double *, sz_med_t length);
static void simplexproj_Duchi(double *y, double *x, const unsigned int length);
static void matvec(double *, double *, double *, int, int);
static double frob_norm(double *, sz_med_t);

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

void k_simplex(double *theta, d_mat_t P, double *g, int max_walk, double lambda, double *dummy, bool rmse)
{

	if (!rmse)
	{
		// adjust for BPR  metric
		for (int i = 1; i < P.num_rows; i++)
		{
			for (int j = 0; j < P.num_cols; j++)
				P.val[i][j] = P.val[0][j] - P.val[i][j];
		}
	}

	// Unfold P for BLAS compatibility
	double *P_unfold = unfold_matrix(P);

	// Prepare parameters for constraned QP

	// first the Hessian matrix
	double A[max_walk * max_walk];
	grammian_matrix(A, P_unfold, (int)P.num_rows, (int)P.num_cols);
	for (int i = 0; i < max_walk; i++)
		A[i * max_walk + i] += lambda + L2_REG_LAMBDA;

	// then the linear part
	double b[max_walk];

	if (rmse)
	{
		for (int i = 0; i < max_walk; i++)
			b[i] = -2.0 * (P_unfold[i] + lambda * g[i]);
	}
	else
	{
		for (int i = 0; i < max_walk; i++)
		{
			b[i] = 0.0;
			for (int j = 0; j < P.num_rows; j++)
				b[i] += P.val[i][j];
			b[i] *= -2.0;
			b[i] += -2.0 * lambda * g[i];
		}
	}

	constr_QP_with_PG(theta, A, b, (sz_med_t)max_walk);

	//free
	free(P_unfold);
}

// Simple search returns the single best step ( i.e. theta = [0 0 0 1 0 0 .... 0 0 ] )

void single_best(double *theta, d_mat_t P, double *g, int max_walk, double lambda, double *dummy, bool rmse)
{

	if (!rmse)
	{
		// adjust for BPR  metric
		for (int i = 1; i < P.num_rows; i++)
		{
			for (int j = 0; j < P.num_cols; j++)
				P.val[i][j] = P.val[0][j] - P.val[i][j];
		}
	}

	for (int k = 0; k < max_walk; k++)
		theta[k] = 0.0;

	double loss[max_walk];

	//Compute loss of each k-step and find smallest
	int best_k = 0;
	double min_loss = 1.0e5 * lambda;
	for (int k = 0; k < max_walk; k++)
	{
		loss[k] = lambda * pow(g[k] - 1.0, 2); // first the regularizer
		loss[k] += pow(1.0 - P.val[0][k], 2);
		if (rmse)
		{
			for (int j = 1; j < P.num_rows; j++)
				loss[k] += P.val[j][k] * P.val[j][k];
		}
		else
		{
			for (int j = 1; j < P.num_rows; j++)
				loss[k] += pow(1.0 - P.val[j][k], 2);
		}
		if (loss[k] < min_loss)
		{
			min_loss = loss[k];
			best_k = k;
		}
	}

	theta[best_k] = 1.0;
}

void ppr_grid(double *theta, d_mat_t P, double *g, int max_walk, double lambda, double *ppr_c, bool rmse)
{

	if (!rmse)
	{
		// adjust for BPR  metric
		for (int i = 1; i < P.num_rows; i++)
		{
			for (int j = 0; j < P.num_cols; j++)
				P.val[i][j] = P.val[0][j] - P.val[i][j];
		}
	}

	// Unfold P for BLAS compatibility
	double *P_unfold = unfold_matrix(P);

	// Candidate diffusions
	double *C = (double *)malloc(P.num_rows * GRID_POINTS * sizeof(double));
	matrix_matrix_product(C, P_unfold, ppr_c, P.num_rows, max_walk, GRID_POINTS);

	// find the best one
	double loss[GRID_POINTS];
	int best_point = 0;
	double min_loss = 1.0e5 * lambda;
	for (int i = 0; i < GRID_POINTS; i++)
	{
		// first the regularizer
		loss[i] = 0.0;
		for (int k = 0; k < max_walk; k++)
			loss[i] += lambda * pow(g[k] - ppr_c[k * GRID_POINTS + i], 2);

		// then the ratings
		if (rmse)
		{
			loss[i] += pow(1.0 - C[i], 2);
			for (int j = 1; j < P.num_rows; j++)
				loss[i] += pow(C[j * GRID_POINTS + i], 2);
		}
		else
		{
			for (int j = 0; j < P.num_rows; j++)
				loss[i] += pow(1.0 - C[j * GRID_POINTS + i], 2);
		}

		if (loss[i] < min_loss)
		{
			min_loss = loss[i];
			best_point = i;
		}
	}

	// return the best one
	for (int k = 0; k < max_walk; k++)
		theta[k] = ppr_c[k * GRID_POINTS + best_point];

	//free
	free(P_unfold);
	free(C);
}

void hk_grid(double *theta, d_mat_t P, double *g, int max_walk, double lambda, double *hk_c, bool rmse)
{

	if (!rmse)
	{
		// adjust for BPR  metric
		for (int i = 1; i < P.num_rows; i++)
		{
			for (int j = 0; j < P.num_cols; j++)
				P.val[i][j] = P.val[0][j] - P.val[i][j];
		}
	}

	// Unfold P for BLAS compatibility
	double *P_unfold = unfold_matrix(P);

	double *C = (double *)malloc(P.num_rows * GRID_POINTS * sizeof(double));
	matrix_matrix_product(C, P_unfold, hk_c, P.num_rows, max_walk, GRID_POINTS);

	// find the best one
	double loss[GRID_POINTS];
	int best_point = 0;
	double min_loss = 1.0e5 * lambda;
	for (int i = 0; i < GRID_POINTS; i++)
	{
		// first the regularizer
		loss[i] = 0.0;
		for (int k = 0; k < max_walk; k++)
			loss[i] += lambda * pow(g[k] - hk_c[k * GRID_POINTS + i], 2);

		// then the ratings
		if (rmse)
		{
			loss[i] += pow(1.0 - C[i], 2);
			for (int j = 1; j < P.num_rows; j++)
				loss[i] += pow(C[j * GRID_POINTS + i], 2);
		}
		else
		{
			for (int j = 0; j < P.num_rows; j++)
				loss[i] += pow(1.0 - C[j * GRID_POINTS + i], 2);
		}

		if (loss[i] < min_loss)
		{
			min_loss = loss[i];
			best_point = i;
		}
	}

	// return the best one
	for (int k = 0; k < max_walk; k++)
		theta[k] = hk_c[k * GRID_POINTS + best_point];

	//free
	free(P_unfold);
	free(C);
}

void dictionary_single(double *theta, d_mat_t P, double *g, int max_walk, double lambda, double *dict, bool rmse)
{

	if (!rmse)
	{
		// adjust for BPR  metric
		for (int i = 1; i < P.num_rows; i++)
		{
			for (int j = 0; j < P.num_cols; j++)
				P.val[i][j] = P.val[0][j] - P.val[i][j];
		}
	}

	// Unfold P for BLAS compatibility
	double *P_unfold = unfold_matrix(P);

	// Prepare matrices
	double *PD = (double *)malloc(P.num_rows * GRID_POINTS * sizeof(double));
	matrix_matrix_product(PD, P_unfold, dict, (int)P.num_rows, (int)P.num_cols, GRID_POINTS);
	double DD[GRID_POINTS * GRID_POINTS];
	grammian_matrix(DD, dict, max_walk, GRID_POINTS);
	double A[GRID_POINTS * GRID_POINTS];
	grammian_matrix(A, PD, (int)P.num_rows, GRID_POINTS);
	for (int i = 0; i < GRID_POINTS * GRID_POINTS; i++)
		A[i] += lambda * DD[i];

	double g_D[GRID_POINTS];
	for (int i = 0; i < GRID_POINTS; i++)
	{
		g_D[i] = 0.0;
		for (int k = 0; k < max_walk; k++)
		{
			g_D[i] += g[k] * dict[k * GRID_POINTS + i];
		}
		g_D[i] *= -2.0 * lambda;
	}

	if (rmse)
	{
		for (int i = 0; i < GRID_POINTS; i++)
			g_D[i] += -2.0 * PD[i];
	}
	else
	{
		for (int i = 0; i < GRID_POINTS; i++)
		{
			for (int j = 0; j < P.num_rows; j++)
				g_D[i] += -2.0 * PD[j * GRID_POINTS + i];
		}
	}

	double theta_dict[GRID_POINTS];

	// call QP solver for dictionary coefficients
	constr_QP_with_PG(theta_dict, A, g_D, (sz_med_t)GRID_POINTS);

	// construct diffusion from dictionary coefficients
	matvec(theta, dict, theta_dict, max_walk, GRID_POINTS);

	// free
	free(PD);
	free(P_unfold);
}


void dictionary(double *theta, d_mat_t P, double *g, int max_walk, double lambda, double *dict, bool rmse)
{

	if (!rmse)
	{
		// adjust for BPR  metric
		for (int i = 1; i < P.num_rows; i++)
		{
			for (int j = 0; j < P.num_cols; j++)
				P.val[i][j] = P.val[0][j] - P.val[i][j];
		}
	}

	// Unfold P for BLAS compatibility
	double *P_unfold = unfold_matrix(P);

	// Prepare matrices
	double *PD = (double *)malloc(P.num_rows * 2 * GRID_POINTS * sizeof(double));
	matrix_matrix_product(PD, P_unfold, dict, (int)P.num_rows, (int)P.num_cols, 2 * GRID_POINTS);
	double DD[4 * GRID_POINTS * GRID_POINTS];
	grammian_matrix(DD, dict, max_walk, 2 * GRID_POINTS);
	double A[4 * GRID_POINTS * GRID_POINTS];
	grammian_matrix(A, PD, (int)P.num_rows, 2 * GRID_POINTS);
	for (int i = 0; i < 4 * GRID_POINTS * GRID_POINTS; i++)
		A[i] += lambda * DD[i];

	double g_D[2 * GRID_POINTS];
	for (int i = 0; i < 2 * GRID_POINTS; i++)
	{
		g_D[i] = 0.0;
		for (int k = 0; k < max_walk; k++)
		{
			g_D[i] += g[k] * dict[k * 2 * GRID_POINTS + i];
		}
		g_D[i] *= -2.0 * lambda;
	}

	if (rmse)
	{
		for (int i = 0; i < 2 * GRID_POINTS; i++)
			g_D[i] += -2.0 * PD[i];
	}
	else
	{
		for (int i = 0; i < 2 * GRID_POINTS; i++)
		{
			for (int j = 0; j < P.num_rows; j++)
				g_D[i] += -2.0 * PD[j * 2 * GRID_POINTS + i];
		}
	}

	double theta_dict[2 * GRID_POINTS];

	// call QP solver for dictionary coefficients
	constr_QP_with_PG(theta_dict, A, g_D, (sz_med_t)2 * GRID_POINTS);

	// construct diffusion from dictionary coefficients
	matvec(theta, dict, theta_dict, max_walk, 2 * GRID_POINTS);

	// free
	free(PD);
	free(P_unfold);
}

// Returns PPR coefficients for given points on a grid of parameter a
double *ppr_coefficients(int max_walk)
{

	int grid_points = GRID_POINTS;

	double *coef = (double *)malloc(max_walk * grid_points * sizeof(double));
	double width = 1.0 / (double)(grid_points + 2);

	double col_sum[grid_points];
	for (int j = 0; j < grid_points; j++)
		col_sum[j] = 0.0;

	for (int i = 0; i < max_walk; i++)
	{
		for (int j = 0; j < grid_points; j++)
		{
			// pagerank weights per step
			coef[i * grid_points + j] = (i == 0) ? (1 - (0.01 + width * ((double)j))) : (1 - (0.01 + width * ((double)j))) * pow(0.01 + width * ((double)j), i);
			col_sum[j] += coef[i * grid_points + j];
		}
	}
	// printf("\nCoef test: %lf , %lf, %lf, %lf\n",coef[0+3],coef[1*grid_points+3],coef[2*grid_points+3],coef[3*grid_points+3]);exit(0);
	// Normalize coefficients so tha they sum to 1 (preserves random-walk interpretability)
	for (int i = 0; i < max_walk; i++)
	{
		for (int j = 0; j < grid_points; j++)
			coef[i * grid_points + j] /= col_sum[j];
	}

	return coef;
}

// Returns HK coefficients for given points on a grid of parameter \tau
double *hk_coefficients(int max_walk)
{

	int grid_points = GRID_POINTS;

	double *coef = (double *)malloc(max_walk * grid_points * sizeof(double));
	double width = (double)MAX_HK_COEF / (double)(grid_points + 2);

	// compute factorials from 1 to max_walk
	sz_long_t factorial[max_walk];
	factorial[0] = 1;
	for (int j = 1; j < max_walk; j++)
		factorial[j] = factorial[j - 1] * j;

	double col_sum[grid_points];
	for (int j = 0; j < grid_points; j++)
		col_sum[j] = 0.0;

	for (int i = 0; i < max_walk; i++)
	{
		for (int j = 0; j < grid_points; j++)
		{
			coef[i * grid_points + j] = pow(1.0 + width * ((double)j), i) / (double)factorial[i];
			col_sum[j] += coef[i * grid_points + j];
		}
	}
	//printf("\nCoef test: %lf , %lf, %lf, %lf\n",coef[0+5],coef[1*grid_points+5],coef[2*grid_points+5],coef[3*grid_points+5]);exit(0);
	// Normalize coefficients so that they sum to 1 (preserves random-walk interpretability)
	for (int i = 0; i < max_walk; i++)
	{
		for (int j = 0; j < grid_points; j++)
			coef[i * grid_points + j] /= col_sum[j];
	}

	return coef;
}

// Returns dictionary coefficients by concatenating PPR and HK
double *dict_coefficients(int max_walk)
{
	int grid_points = GRID_POINTS;

	double *coef = (double *)malloc(2 * max_walk * grid_points * sizeof(double));

	double *hk = hk_coefficients(max_walk);
	double *ppr = ppr_coefficients(max_walk);

	for (int i = 0; i < max_walk; i++)
	{
		for (int j = 0; j < grid_points; j++)
		{
			coef[2 * i * grid_points + j] = ppr[i * grid_points + j];
		}
		for (int j = grid_points; j < 2 * grid_points; j++)
		{
			coef[2 * i * grid_points + j] = hk[i * grid_points + j - grid_points];
		}
	}

	//free temporary coefs
	free(ppr);
	free(hk);
	return coef;
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

// AUXILIARY FUNCTIONS

//Interface for CBLAS mutrix matrix product
// C =A*B
// A : m x k
// B : k x n
static void matrix_matrix_product(double *C, double *A, double *B, int m, int k, int n)
{

	for (int i = 0; i < m * n; i++)
		C[i] = 0.0f;

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,
				n, k, 1.0f, A, k, B, n, 0.0f, C, n);
}

// Unfold matrix from A[i[j] to A[i*M + j]
static double *unfold_matrix(d_mat_t A)
{

	double *A_u = (double *)malloc(A.num_cols * A.num_rows * sizeof(double));

	for (sz_long_t i = 0; i < A.num_rows; i++)
		memcpy(A_u + i * A.num_cols, A.val[i], A.num_cols * sizeof(double));

	return A_u;
}

//Solving constrained quadratic minimization via projected gradient descent
// The following function returns x =arg min {x^T*A*x +x^T*B} s.t. x in Prob. Simplex
static void constr_QP_with_PG(double *x, double *A, double *b, sz_med_t K)
{
	double inf_norm, step_size;
	double x_temp[K];
	double x_prev[K];

	step_size = STEPSIZE_2 / frob_norm(A, K);

	//Initialize to uniform
	for (sz_med_t i = 0; i < K; i++)
		x[i] = 1.0f / (double)K;

	sz_med_t iter = 0;
	memcpy(x_prev, x, K * sizeof(double));
	// int count=0;
	do
	{
		iter++;
		//step_size = 1 / ((double)iter) * step_size;
		//step_size = 1/(sqrt(iter));//*step_size;
		//step_size = 1/(double)iter;
		//Take gradient step
		matvec(x_temp, A, x, K, K);

		for (sz_med_t j = 0; j < K; j++)
			x[j] -= step_size * (2.0f * x_temp[j] + b[j]);

		//project to feasible set
		simplexproj_Duchi(x, x, K);

#if DEBUG
		printf("\n COST: ");
		printf(" %lf ", cost_func(A, b, x, K));
#endif

		inf_norm = max_abs_dif(x_prev, x, (sz_long_t)K);

		memcpy(x_prev, x, K * sizeof(double));

	} while (iter < MAXIT_GD && inf_norm > GD_TOL_2);

	// printf("\n Counter = %d", count);
	// printf("\n Optimization finished after: %"PRIu32" iterations\n", (uint32_t) iter);
}

// Grammian matrix G =A'*A using CBLAS
// A : m x n
static void grammian_matrix(double *G, double *A, int m, int n)
{

	for (int i = 0; i < n * n; i++)
		G[i] = 0.0f;

	double *A_copy = (double *)malloc(m * n * sizeof(double));

	memcpy(A_copy, A, m * n * sizeof(double));

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n,
				n, m, 1.0f, A_copy, n, A, n, 0.0f, G, n);

	free(A_copy);
}

//Interface for CBLAS matrix vector product
// A : M x N
static void matvec(double *y, double *A, double *x, int M, int N)
{

	// for (int i = 0; i < M; i++)
	// {
	// 	y[i] = 0.0f;
	// }

	cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, A, N, x, 1, 0.0f, y, 1);
}

// 

/* Algorithm using partitioning with respect to a pivot, chosen randomly, 
as given by Duchi et al. in "Efficient Projections onto the l1-Ball for 
Learning in High Dimensions" */
static void simplexproj_Duchi(double *y, double *x, const unsigned int length)
{
	double *auxlower = (x == y ? (double *)malloc(length * sizeof(double)) : x);
	double *auxupper = (double *)malloc(length * sizeof(double));
	double *aux = auxlower;
	double pivot;
	int auxlowerlength = 0;
	int auxupperlength = 1;
	int upperlength;
	int auxlength;
	int i = 0;
	int pospivot = (int)(rand() / (((double)RAND_MAX + 1.0) / length));
	double tauupper;
	double tau = (pivot = y[pospivot]) - 1.0;
	while (i < pospivot)
		if (y[i] < pivot)
			auxlower[auxlowerlength++] = y[i++];
		else
		{
			auxupper[auxupperlength++] = y[i];
			tau += (y[i++] - tau) / auxupperlength;
		}
	i++;
	while (i < length)
		if (y[i] < pivot)
			auxlower[auxlowerlength++] = y[i++];
		else
		{
			auxupper[auxupperlength++] = y[i];
			tau += (y[i++] - tau) / auxupperlength;
		}
	if (tau < pivot)
	{
		upperlength = auxupperlength;
		tauupper = tau;
		auxlength = auxlowerlength;
	}
	else
	{
		tauupper = 0.0;
		upperlength = 0;
		aux = auxupper + 1;
		auxlength = auxupperlength - 1;
	}
	while (auxlength > 0)
	{
		pospivot = (int)(rand() / (((double)RAND_MAX + 1.0) / auxlength));
		if (upperlength == 0)
			tau = (pivot = aux[pospivot]) - 1.0;
		else
			tau = tauupper + ((pivot = aux[pospivot]) - tauupper) / (upperlength + 1);
		i = 0;
		auxlowerlength = 0;
		auxupperlength = 1;
		while (i < pospivot)
			if (aux[i] < pivot)
				auxlower[auxlowerlength++] = aux[i++];
			else
			{
				auxupper[auxupperlength++] = aux[i];
				tau += (aux[i++] - tau) / (upperlength + auxupperlength);
			}
		i++;
		while (i < auxlength)
			if (aux[i] < pivot)
				auxlower[auxlowerlength++] = aux[i++];
			else
			{
				auxupper[auxupperlength++] = aux[i];
				tau += (aux[i++] - tau) / (upperlength + auxupperlength);
			}
		if (tau < pivot)
		{
			upperlength += auxupperlength;
			tauupper = tau;
			auxlength = auxlowerlength;
			aux = auxlower;
		}
		else
		{
			aux = auxupper + 1;
			auxlength = auxupperlength - 1;
		}
	}
	for (i = 0; i < length; i++)
		x[i] = (y[i] > tau ? y[i] - tauupper : 0.0000000001);
	double sum = 0.0;
	for (i = 0; i < length; i++)
		sum += x[i];
	for (i = 0; i < length; i++)
		x[i] /= sum;
	if (x == y)
		free(auxlower);
	free(auxupper);
}


//Infinity norm
static double max_abs_dif(double *a, double *b, sz_long_t len)
{
	double dif, max = 0.0;

	for (sz_long_t i = 0; i < len; i++)
	{
		dif = fabs(a[i] - b[i]);
		max = (dif > max) ? dif : max;
	}

	return max;
}

#if DEBUG
//Evaluates quadratic with Hessian A and linear part b at x
static double cost_func(double *A, double *b, double *x, sz_med_t len)
{

	double quad = 0.0f, lin = 0.0f;

	for (sz_med_t i = 0; i < len; i++)
	{
		for (sz_med_t j = 0; j < len; j++)
		{
			quad += A[i * len + j] * x[i] * x[j];
		}
		lin += b[i] * x[i];
	}
	return quad + lin;
}
#endif

//frobenious norm of double-valued square matrix
static double frob_norm(double *A, sz_med_t dim)
{
	double norm = 0.0f;

	for (sz_med_t i = 0; i < dim * dim; i++)
	{
		norm += pow(A[i], 2.0f);
	}

	return sqrt(norm);
}
