////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains high-level routines for parsing the command line for arguments,
 and handling multiclass or multilabel input.


 Code by: Dimitris Berberidis and Athanasios N. Nikolakopoulos
 University of Minnesota 2019
*/

///////////////////////////////////////////////////////////////////////////////////////////////

#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rec_defs.h"
#include "rec_mem.h"

// Alocate memory for csr
void csr_alloc(csr_graph_t *graph) {
  sz_long_t nnz_buff = 500 * ITM_BUFF_SIZE, num_nodes_buff = ITM_BUFF_SIZE;

  graph->csr_value = (long double *)malloc(nnz_buff * sizeof(long double));
  graph->csr_column = (sz_long_t *)malloc(nnz_buff * sizeof(sz_long_t));
  graph->csr_row_pointer =
      (sz_long_t *)malloc(num_nodes_buff * sizeof(sz_long_t));
  graph->degrees = (sz_long_t *)malloc(num_nodes_buff * sizeof(sz_long_t));
}

// Reallocate csr memory after size is known
void csr_realloc(csr_graph_t *graph, sz_long_t nnz, sz_long_t num_nodes) {
  graph->csr_value = realloc(graph->csr_value, nnz * sizeof(long double));
  graph->csr_column = realloc(graph->csr_column, nnz * sizeof(sz_long_t));
  graph->csr_row_pointer =
      realloc(graph->csr_row_pointer, (num_nodes + 1) * sizeof(sz_long_t));
  graph->degrees = realloc(graph->degrees, num_nodes * sizeof(sz_long_t));
  graph->num_nodes = num_nodes;
  graph->nnz = nnz;
}

// Free memory allocated to csr_graph
void csr_free(csr_graph_t graph) {
  free(graph.csr_value);
  free(graph.csr_column);
  free(graph.csr_row_pointer);
  free(graph.degrees);
}

// Free memory allocated to array of csr_graphs
void csr_array_free(csr_graph_t *graph_array, sz_short_t num_copies) {
  for (sz_short_t i = 0; i < num_copies; i++)
    csr_free(graph_array[i]);
  free(graph_array);
}

// Allocate double matrix
d_mat_t d_mat_init(sz_long_t num_rows, sz_long_t num_cols) {
  d_mat_t mat = {.num_rows = num_rows, .num_cols = num_cols};
  mat.val = (double **)malloc(num_rows * sizeof(double *));
  for (sz_long_t i = 0; i < num_rows; i++)
    mat.val[i] = (double *)malloc(num_cols * sizeof(double));

  if (mat.val[num_rows - 1] == NULL)
    printf(" ERROR: at matrix allocation \n");

  return mat;
}

// Resize double matrix
void d_mat_resize(d_mat_t *mat, sz_long_t new_num_rows,
                  sz_long_t new_num_cols) {
  sz_long_t num_rows = mat->num_rows;
  if (new_num_rows > num_rows) {
    for (sz_long_t i = 0; i < num_rows; i++)
      mat->val[i] =
          (double *)realloc(mat->val[i], new_num_cols * sizeof(double));
    for (sz_long_t i = num_rows; i < new_num_rows; i++)
      mat->val[i] = (double *)malloc(new_num_cols * sizeof(double));
  } else {
    for (sz_long_t i = new_num_rows; i < num_rows; i++)
      free(mat->val[i]);
    for (sz_long_t i = 0; i < new_num_rows; i++)
      mat->val[i] =
          (double *)realloc(mat->val[i], new_num_cols * sizeof(double));
  }
  mat->val = (double **)realloc(mat->val, new_num_rows * sizeof(double *));

  mat->num_rows = new_num_rows;
  mat->num_cols = new_num_cols;
}

// Free double rating_matrix_file
void d_mat_free(d_mat_t mat) {
  for (sz_long_t i = 0; i < mat.num_rows; i++)
    free(mat.val[i]);
  free(mat.val);
}

// Allocate integer matrix
i_mat_t i_mat_init(sz_long_t num_rows, sz_long_t num_cols) {
  i_mat_t mat = {.num_rows = num_rows, .num_cols = num_cols};
  mat.val = (sz_long_t **)malloc(num_rows * sizeof(sz_long_t *));
  for (sz_long_t i = 0; i < num_rows; i++)
    mat.val[i] = (sz_long_t *)malloc(num_cols * sizeof(sz_long_t));

  if (mat.val[num_rows - 1] == NULL)
    printf(" ERROR: at matrix allocation \n");

  return mat;
}

// Allocate integer matrix
void i_mat_resize(i_mat_t *mat, sz_long_t new_num_rows,
                  sz_long_t new_num_cols) {
  sz_long_t num_rows = mat->num_rows;
  if (new_num_rows > num_rows) {
    for (sz_long_t i = 0; i < num_rows; i++)
      mat->val[i] =
          (sz_long_t *)realloc(mat->val[i], new_num_cols * sizeof(sz_long_t));
    for (sz_long_t i = num_rows; i < new_num_rows; i++)
      mat->val[i] = (sz_long_t *)malloc(new_num_cols * sizeof(sz_long_t));
  } else {
    for (sz_long_t i = new_num_rows; i < num_rows; i++)
      free(mat->val[i]);
    for (sz_long_t i = 0; i < new_num_rows; i++)
      mat->val[i] =
          (sz_long_t *)realloc(mat->val[i], new_num_cols * sizeof(sz_long_t));
  }
  mat->val =
      (sz_long_t **)realloc(mat->val, new_num_rows * sizeof(sz_long_t *));

  mat->num_rows = new_num_rows;
  mat->num_cols = new_num_cols;
}

// Free integer rating_matrix_file
void i_mat_free(i_mat_t mat) {
  for (sz_long_t i = 0; i < mat.num_rows; i++)
    free(mat.val[i]);
  free(mat.val);
}

// free array of users (rating matrix)
void usr_free(user_t *users, sz_long_t num_users) {
  for (sz_long_t i = 0; i < num_users; i++)
    free(users[i].items);
  free(users);
}

// Initialize and allocate buffers for input data
data_t data_init(void) {

  data_t data;

  data = (data_t){
      .no_models = false,
      .cv_neg = i_mat_init(ITM_BUFF_SIZE, USR_BUFF_SIZE),
      .cv_pos = (sz_long_t *)malloc(USR_BUFF_SIZE * sizeof(sz_long_t)),
      .test_neg = i_mat_init(ITM_BUFF_SIZE, USR_BUFF_SIZE),
      .test_pos = (sz_long_t *)malloc(USR_BUFF_SIZE * sizeof(sz_long_t))};

  return data;
}

// Initialize and allocate buffers for input data
data_t datacv_init(void) {

  data_t data;

  data = (data_t){
               .no_models = false,
               .cv_neg = i_mat_init(ITM_BUFF_SIZE, USR_BUFF_SIZE),
               .cv_pos = (sz_long_t *)malloc(USR_BUFF_SIZE * sizeof(sz_long_t)),
               .test_neg = NULL,
               .test_pos = NULL};

  return data;
}

// free input data
void data_free(data_t data) {
  csr_array_free(data.item_models, data.num_models);
  usr_free(data.users, data.num_users);
  if (data.users_full != NULL)
    usr_free(data.users_full, data.num_users);
  free(data.cv_pos);
  i_mat_free(data.cv_neg);
}

// allocate output data
out_t output_init(data_t data, ctrl_t ctrl) {
  out_t output;
  output.num_models = data.num_models;
  output.num_users = data.num_users;
  output.thetas = (d_mat_t *)malloc(data.num_models * sizeof(d_mat_t));
  output.mus =
      (d_mat_t *)malloc(data.num_models * sizeof(d_mat_t)); // added for mus
  output.general_trends =
      (metric_t **)malloc(data.num_models * sizeof(metric_t *));
  output.test_result =
      (metric_t **)malloc(data.num_models * sizeof(metric_t *));

  for (int i = 0; i < data.num_models; i++) {
    output.thetas[i] = d_mat_init(data.num_users, ctrl.max_walk);
    output.mus[i] =
        d_mat_init(data.num_users, ctrl.max_walk - 1); // mus are less than θs
    output.general_trends[i] =
        (metric_t *)malloc(ctrl.max_walk * sizeof(metric_t));
    output.test_result[i] =
        (metric_t *)malloc(data.num_users * sizeof(metric_t));
    for (int k = 0; k < ctrl.max_walk; k++)
      output.general_trends[i][k] =
          (metric_t){.HR = 0.0, .ARHR = 0.0, .NDCG = 0.0};
  }

  if (ctrl.save_vals) {
    output.val_pred = (d_mat_t *)malloc(data.num_models * sizeof(d_mat_t));
    output.test_pred = (d_mat_t *)malloc(data.num_models * sizeof(d_mat_t));
    for (int i = 0; i < data.num_models; i++) {
      output.val_pred[i] = d_mat_init(data.num_users, data.cv_neg.num_rows + 1);
      output.test_pred[i] =
          d_mat_init(data.num_users, data.test_neg.num_rows + 1);
    }
  }

  output.local_trend =
      init_local_trend(ctrl.usr_threads, data.num_models, ctrl.max_walk);

  return output;
}
// free output data
void output_free(out_t output, ctrl_t ctrl) {

  for (int i = 0; i < output.num_models; i++) {
    d_mat_free(output.thetas[i]);
    d_mat_free(output.mus[i]);
    free(output.general_trends[i]);
    free(output.test_result[i]);
  }
  free(output.thetas);
  free(output.mus);
  free(output.general_trends);
  free(output.test_result);

  if (ctrl.save_vals) {
    for (int i = 0; i < output.num_models; i++) {
      d_mat_free(output.val_pred[i]);
      d_mat_free(output.test_pred[i]);
    }
    free(output.val_pred);
    free(output.test_pred);
  }

  free_local_trend(output.local_trend, ctrl.usr_threads, output.num_models);
}

// allocate and initialize local trends
metric_t ***init_local_trend(int N, int M, int K) {
  metric_t ***lt = (metric_t ***)malloc(N * sizeof(metric_t **));
  for (int i = 0; i < N; i++) {
    lt[i] = (metric_t **)malloc(M * sizeof(metric_t *));
    for (int j = 0; j < M; j++) {
      lt[i][j] = (metric_t *)malloc(K * sizeof(metric_t));
      for (int k = 0; k < K; k++)
        lt[i][j][k] = (metric_t){.HR = 0.0, .ARHR = 0.0, .NDCG = 0.0};
    }
  }
  return lt;
}

// free local trends
void free_local_trend(metric_t ***lt, int N, int M) {

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++)
      free(lt[i][j]);
    free(lt[i]);
  }
  free(lt);
}