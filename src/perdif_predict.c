///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains the PERDIF program that implements personalized diffusions for
 recommendations.

 Code by: Dimitris Berberidis and Athanasios N. Nikolakopoulos
 University of Minnesota 2019
*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "csr_handling.h"
#include "perdif_mthreads.h"
#include "perdif_predict.h"
#include "rec_IO.h"
#include "rec_defs.h"
#include "rec_graph_gen.h"
#include "rec_mem.h"
#include "rec_metrics.h"

// Function that uses R_full and applies learnt thetas and item models
void perdif_predict(data_t data, out_t out, ctrl_t ctrl, output_t outfiles) {

  // Create num_models X graphs (using matrix_full) to be worked on in parallel
  csr_graph_t *graphs =
      (csr_graph_t *)malloc(data.num_models * sizeof(csr_graph_t));
  for (int i = 0; i < data.num_models; i++) {
    if (VERBOSE)
      printf("Building TestSet Graph: %d out of %d \n", i + 1, data.num_models);
    graphs[i] = generate_rec_graph(data.users_full, data.item_models[i],
                                   data.num_users, ctrl.bipartite);
  }

  // Testing Top-N Recommendation Performance
  green();
  printf("\nTesting Top-N Recommendation Performance...\n");
  colorreset();
  distribute_to_threads(graphs, data, out, ctrl, &perdif_predict_smodel);

  // Also store the name of the item_model
  const char *names[data.num_models];
  for (int i = 0; i < data.num_models; i++)
    names[i] = data.item_models[i].name;

  store_results(out.test_result, outfiles.outdir_pred, data.num_models,
                data.num_users, names);

  if (ctrl.bipartite)
    store_vals(outfiles.outdir_pred, out.test_pred, data.num_models, false);

  // free memory
  csr_array_free(graphs, data.num_models);
}

// test-ratings prediction for a single model and window of users
void perdif_predict_smodel(struct pass2thread_t thread_data,
                           int model_counter) {

  // Unpack thread data
  // d_mat_t pred;
  metric_t *result = thread_data.out.test_result[model_counter];
  d_mat_t thetas = thread_data.out.thetas[model_counter];
  // csr_graph_t graph = thread_data.graphs[model_counter][thread_data.u_id];
  csr_graph_t graph = thread_data.graphs[model_counter][0];
  user_t *users = thread_data.data.users_full;
  i_mat_t test_neg = thread_data.data.test_neg;
  sz_long_t *test_pos = thread_data.data.test_pos;
  ctrl_t ctrl = thread_data.ctrl;
  sz_long_t from_user = thread_data.usr_start;
  sz_long_t num_users = thread_data.usr_win;
  sz_long_t global_num_users = thread_data.data.num_users;

  report_t report = thread_data.report;

  // Allocate landing prob vectors
  double *p = (double *)malloc(graph.num_nodes * sizeof(double));
  double *p_plus = (double *)malloc(graph.num_nodes * sizeof(double));

  // For storage of tes tvalues
  double p_pos;
  double p_neg[test_neg.num_rows];

  // percentage unit used for progress report
  int one_per_cent = num_users / 100;

  // Iterate over all users
  for (sz_long_t i = from_user; i < from_user + num_users; i++) {

    p_pos = 0.0;
    for (int j = 0; j < test_neg.num_rows; j++)
      p_neg[j] = 0.0;

    // Copy user cv ratings to minimize read-conflicts with other threads
    sz_long_t pos = test_pos[i];
    sz_long_t neg[test_neg.num_rows];
    for (int j = 0; j < test_neg.num_rows; j++)
      neg[j] = test_neg.val[j][i];

    // Initialize land prob vector for user i
    for (sz_long_t j = 0; j < graph.num_nodes; j++) {
      p[j] = 0.0;
      p_plus[j] = 0.0;
    }

    if (ctrl.bipartite) {
      p[i] = 1.0;
    } else {
      for (int j = 0; j < users[i].num_items; j++)
        p[users[i].items[j]] = 1.0 / (double)users[i].num_items;
    }

    // Iterate through every step of walk
    for (int k = 0; k < ctrl.max_walk; k++) {
      if (k == 0)
        continue;
      // Advance one step
      my_CSR_matvec(p_plus, p, graph);

      // Build p_pos and p_neg using learnt parameters
      sz_long_t offset = (ctrl.bipartite) ? global_num_users : 0;
      p_pos += p_plus[pos + offset] * thetas.val[i][k];
      for (int j = 0; j < test_neg.num_rows; j++)
        p_neg[j] += p_plus[neg[j] + offset] * thetas.val[i][k];

      // Swap pointers
      double *temp = p;
      p = p_plus;
      p_plus = temp;
    }

    // Possibly save values
    if (ctrl.save_vals) {
      d_mat_t pred = thread_data.out.test_pred[model_counter];
      pred.val[i][0] = p_pos;
      for (int j = 1; j < pred.num_cols; j++) {
        pred.val[i][j] = p_neg[j];
      }
    }

    // Evaluate user on test ratings
    result[i] = get_rec_metrics(p_pos, p_neg, test_neg.num_rows);

    // report progress (estimate from one thread)
    if (report.yes && ((i - from_user) % (one_per_cent * report.every) == 0))
      if (one_per_cent > 0) {
        if (i == from_user)
        // printf(" testing %d %% complete\n ", report.shift + (int)(i -
        // from_user) / (one_per_cent * report.every));
        {
          printf("Progress: ");
          fflush(stdout);
        } else {
          printf("|");
          fflush(stdout);
        }
      }
    if (report.yes && i == from_user + num_users - 1) {
      printf(" Done!\n");
      fflush(stdout);
      colorreset();
    }
  }

  // free
  free(p);
  free(p_plus);
}
