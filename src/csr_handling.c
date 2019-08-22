///////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains routines for handling compressed-sparse-row (CSR) graphs
 ( allocating , copying, normalizing, scaling, mat-vec, mat-mat, freeing )

 Code by: Dimitris Berberidis and Athanasios N. Nikolakopoulos
 University of Minnesota 2019
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <stdbool.h>

#include "csr_handling.h"
#include "rec_defs.h"
#include "rec_mem.h"

static int compare(const void *pa, const void *pb);

// Make a copy of graph with edges multiplied by some scalar
csr_graph_t csr_deep_copy_and_scale(csr_graph_t graph, double scale)
{

	csr_graph_t graph_temp;

	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted.
	//However they will be weighted as sparse stochastic matrices so values will be needed
	graph_temp.csr_value = (long double *)malloc(graph.nnz * sizeof(long double));

	graph_temp.csr_column = (sz_long_t *)malloc(graph.nnz * sizeof(sz_long_t));

	graph_temp.csr_row_pointer = (sz_long_t *)malloc((graph.num_nodes + 1) * sizeof(sz_long_t));

	graph_temp.degrees = (sz_long_t *)malloc(graph.num_nodes * sizeof(sz_long_t));

	graph_temp.num_nodes = graph.num_nodes;

	graph_temp.nnz = graph.nnz;

	//copy data

	memcpy(graph_temp.csr_row_pointer, graph.csr_row_pointer, (graph.num_nodes + 1) * sizeof(sz_long_t));

	memcpy(graph_temp.degrees, graph.degrees, graph.num_nodes * sizeof(sz_long_t));

	memcpy(graph_temp.csr_column, graph.csr_column, graph.nnz * sizeof(sz_long_t));

	for (sz_long_t i = 0; i < graph.nnz; i++)
	{
		graph_temp.csr_value[i] = scale * graph.csr_value[i];
	}

	return graph_temp;
}

// Make a copy of graph with edges multiplied by some scalar
csr_graph_t csr_deep_copy(csr_graph_t graph)
{

	csr_graph_t graph_temp;

	//CSR matrix with three arrays, first is basically a dummy for now since networks I have are unweighted.
	//However they will be weighted as sparse stochastic matrices so values will be needed
	graph_temp.csr_value = (long double *)malloc(graph.nnz * sizeof(long double));

	graph_temp.csr_column = (sz_long_t *)malloc(graph.nnz * sizeof(sz_long_t));

	graph_temp.csr_row_pointer = (sz_long_t *)malloc((graph.num_nodes + 1) * sizeof(sz_long_t));

	graph_temp.degrees = (sz_long_t *)malloc(graph.num_nodes * sizeof(sz_long_t));

	graph_temp.num_nodes = graph.num_nodes;

	graph_temp.nnz = graph.nnz;

	//copy data

	memcpy(graph_temp.csr_row_pointer, graph.csr_row_pointer, (graph.num_nodes + 1) * sizeof(sz_long_t));

	memcpy(graph_temp.degrees, graph.degrees, graph.num_nodes * sizeof(sz_long_t));

	memcpy(graph_temp.csr_column, graph.csr_column, graph.nnz * sizeof(sz_long_t));

	memcpy(graph_temp.csr_value, graph.csr_value, graph.nnz * sizeof(long double));

	return graph_temp;
}

//Return an array with multiple copies of the input graph
csr_graph_t *csr_mult_deep_copy(csr_graph_t graph, sz_short_t num_copies)
{
	csr_graph_t *graph_array = (csr_graph_t *)malloc(num_copies * sizeof(csr_graph_t));
	for (sz_short_t i = 0; i < num_copies; i++)
	{
		graph_array[i] = csr_deep_copy(graph);
	}
	return graph_array;
}

//Subroutine: modify csr_value to be column stochastic
//First find degrees by summing element of each row
//Then go through values and divide by corresponding degree 
//WARNING: Only works for BINARY and SYMMETRIC graph
void make_BCSR_col_stoch(csr_graph_t *graph)
{
	for (sz_long_t i = 0; i < graph->nnz; i++)
	{
		graph->csr_value[i] = graph->csr_value[i] / (long double)graph->degrees[graph->csr_column[i]];
	}
}

//Subroutine: take x, multiply with csr matrix from right and store result in y
void my_CSR_matvec(double *y, double *x, csr_graph_t graph)
{
	// printf("You shouldn't be reading this\n");
	// potential room for improvement using sparse BLAS
	for (sz_long_t i = 0; i < graph.num_nodes; i++)
		y[i] = 0.0;

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		for (sz_long_t j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
			y[i] += x[graph.csr_column[j]] * graph.csr_value[j];
	}
}

//Subroutine: take X, multiply with csr matrix from right and store result in Y
void my_CSR_matmat(double *Y, double *X, csr_graph_t graph, sz_med_t M, sz_med_t from, sz_med_t to)
{

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		for (sz_long_t j = from; j < to; j++)
		{
			Y[i * M + j] = 0.0f;
		}
	}

	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		for (sz_long_t j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
		{
			for (sz_med_t k = from; k < to; k++)
			{
				Y[i * M + k] += X[M * graph.csr_column[j] + k] * graph.csr_value[j];
			}
		}
	}
}

//Transpose csr matrix
csr_graph_t csr_transpose(csr_graph_t graph)
{

	//First expand csr to simple edgelist
	long double *weights;
	sz_long_t **edge_list = csr_to_edgelist(graph, &weights);

	//Flip edgelist
	sz_long_t *alias;
	alias = edge_list[0];
	edge_list[0] = edge_list[1];
	edge_list[1] = alias;

	//Rebuild csr from flipped edgelist
	csr_graph_t graph_trans = edgelist_to_csr(edge_list, weights, graph.num_nodes, graph.nnz);

	//free edgelist
	for (sz_long_t i = 0; i < 2; i++)
		free(edge_list[i]);
	free(edge_list);
	free(weights);
	return graph_trans;
}

//Convert csr to edgelist
//Edge list includes node enumeration [node_a node_b edge_index ]
sz_long_t **csr_to_edgelist(csr_graph_t graph, long double **weights)
{

	//allocate
	sz_long_t **edge_list = (sz_long_t **)malloc(2 * sizeof(sz_long_t *));
	for (sz_long_t i = 0; i < 2; i++)
		edge_list[i] = (sz_long_t *)malloc(graph.nnz * sizeof(sz_long_t));
	*weights = (long double *)malloc(graph.nnz * sizeof(long double));

	//unfold graph
	for (sz_long_t i = 0; i < graph.num_nodes; i++)
	{
		for (sz_long_t j = graph.csr_row_pointer[i]; j < graph.csr_row_pointer[i + 1]; j++)
		{
			edge_list[0][j] = i;
			edge_list[1][j] = graph.csr_column[j];
			*(*weights + j) = graph.csr_value[j];
		}
	}

	return edge_list;
}

//Convert edgelist to csr
csr_graph_t edgelist_to_csr(sz_long_t **edge_list, long double *weights,
							sz_long_t num_nodes, sz_long_t nnz)
{
	csr_graph_t graph;
	csr_alloc(&graph);
	csr_realloc(&graph, nnz, num_nodes);

	//Sort everything with respect to index of first node in every edge
	sz_long_t *temp_ind_b = (sz_long_t *)malloc(nnz * sizeof(sz_long_t));
	sz_long_t *temp_ind_a = (sz_long_t *)malloc(nnz * sizeof(sz_long_t));	
	long double *temp_weights = (long double *)malloc(nnz * sizeof(long double));
	sz_long_t **temp_array = (sz_long_t **)malloc(nnz * sizeof(sz_long_t *));
	for (sz_long_t i = 0; i < nnz; i++)
	{
		temp_array[i] = (sz_long_t *)malloc(2 * sizeof(sz_long_t));
		temp_array[i][0] = edge_list[0][i];
		temp_array[i][1] = i;
	}

	qsort(temp_array, nnz, sizeof(temp_array[0]), compare);

	for (sz_long_t i = 0; i < nnz; i++)
	{
		temp_ind_a[i] = temp_array[i][0]; //these are sorted
		temp_ind_b[i] = edge_list[1][temp_array[i][1]];
		temp_weights[i] = weights[temp_array[i][1]];
	}

	//Fold sorted lists to csr graph
    memcpy(graph.csr_value, temp_weights, nnz * sizeof(long double));
    memcpy(graph.csr_column, temp_ind_b, nnz * sizeof(sz_long_t));

	for(int i =0; i<temp_ind_a[0]+1;i++)
		graph.csr_row_pointer[i] = 0;
	sz_long_t node_count = temp_ind_a[0]; 
	for (sz_long_t i = 0; i < nnz-1; i++){
		sz_long_t dif = temp_ind_a[i+1] - temp_ind_a[i];
		if(dif){
			for(int j=node_count; j< node_count + dif; j++) 
				graph.csr_row_pointer[ j + 1] = i +1;
			node_count += dif; 	
		}
	}

	for(sz_long_t i = node_count + 1; i< num_nodes+1; i++)
		graph.csr_row_pointer[i] = nnz;	

	if(node_count+1 != num_nodes) 
		printf("Warning: collumns missing from model\n");

	for (sz_long_t i = 0; i < num_nodes; i++)
			graph.degrees[i] = graph.csr_row_pointer[i+1] - graph.csr_row_pointer[i];

	
	//free temporary workspace
	for (sz_long_t i = 0; i < nnz; i++)
		free(temp_array[i]);
	free(temp_array);
	free(temp_ind_b);
	free(temp_ind_a);	
	free(temp_weights);
	return graph;
}

//Stochastic normalization of item model acording to max ratings
void csr_rec_normalize(csr_graph_t* graph){

	//Compute sums of columns
	double* col_sums = (double*) malloc(graph->num_nodes*sizeof(double));

	for(sz_long_t i=0;i<graph->num_nodes; i++ ) col_sums[i] =0.0;

	for(sz_long_t i=0;i<graph->nnz; i++ ){
		col_sums[graph->csr_column[i]] += graph->csr_value[i];
	}

	//Divide all entries by max sum
	double max=0.0;
	for(sz_long_t i=0;i<graph->num_nodes; i++ )
		max = (col_sums[i]>max) ? col_sums[i] : max ; 

	for(sz_long_t i=0;i<graph->nnz; i++ )
		graph->csr_value[i] /= max;

	//Compute and add diagonal
	double* diag = (double*) malloc(graph->num_nodes*sizeof(double));
	for(sz_long_t i=0;i<graph->num_nodes; i++ )
		diag[i] = 1.0 - col_sums[i]/max;

	csr_add_diagonal(graph, diag);

	//free
	free(col_sums);
	free(diag);
}

//allccate space and add diagonal array to csr matrix
void csr_add_diagonal(csr_graph_t* graph, double* diag ){

	csr_graph_t graph_temp = csr_deep_copy(*graph);

	csr_realloc(graph, graph->nnz + graph->num_nodes, graph->num_nodes );
	
	graph->csr_row_pointer[0] = 0; 
	for(sz_long_t i=0; i<graph->num_nodes; i++){
		graph->csr_row_pointer[i+1] = graph_temp.csr_row_pointer[i+1] + i +1 ;
		graph->degrees[i] = graph_temp.degrees[i] + 1;
		//printf("\n %d  %d \n", (int) i , (int) graph->degrees[i]);
		for(sz_long_t j=0; j< graph->degrees[i]-1; j++){
			graph->csr_column[graph->csr_row_pointer[i]+j] = graph_temp.csr_column[graph_temp.csr_row_pointer[i]+j];
		    graph->csr_value[graph->csr_row_pointer[i]+j] = graph_temp.csr_value[graph_temp.csr_row_pointer[i]+j];	
		}
		graph->csr_column[graph->csr_row_pointer[i]+graph->degrees[i]-1] = i;
		graph->csr_value[graph->csr_row_pointer[i]+graph->degrees[i]-1] = diag[i];	
	} 

	csr_free( graph_temp );
}


//comparator
static int compare(const void *pa, const void *pb)
{
	const sz_long_t *a = *(const sz_long_t **)pa;
	const sz_long_t *b = *(const sz_long_t **)pb;
	if (a[0] == b[0])
		return a[1] - b[1];
	else
		return a[0] - b[0];
}
