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
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "csr_handling.h"
#include "perdif_fit.h"
#include "perdif_mthreads.h"
#include "perdif_train.h"
#include "rec_IO.h"
#include "rec_defs.h"
#include "rec_graph_gen.h"
#include "rec_mem.h"
#include "rec_metrics.h"

// List of possible methods for fitting diffusion parameters theta
void (*fit_theta[NUM_METHODS])(double *theta, d_mat_t P, double *g,
                               int max_walk, double lambda, double *coef,
                               bool rmse) = {&k_simplex, &single_best,
                                             &dictionary_single,
                                             &dictionary_single, &dictionary};
static void set_trends(out_t out, ctrl_t ctrl);
static void gather_local_trends(out_t, ctrl_t);
void general_trend_smodel(struct pass2thread_t, int);
void perdif_learn_smodel(struct pass2thread_t, int);

// Function that finds the max_walk and the best item_model
void perdif_mselect(data_t data, out_t out, ctrl_t ctrl, output_t outfiles) {
  // Generate model graphs
  csr_graph_t *graphs =
      (csr_graph_t *)malloc(data.num_models * sizeof(csr_graph_t));
  for (int i = 0; i < data.num_models; i++) {
    if (VERBOSE)
      printf("Building the training graphs: %d out of %d \n", i + 1,
             data.num_models);
    graphs[i] = generate_rec_graph(data.users, data.item_models[i],
                                   data.num_users, ctrl.bipartite);
  }

  if (!ctrl.set_trend) {
    // Get general trends in parralel
    yellow();
    printf("\nCross-Validation for the Selection of Item-Model and Max "
           "Walk-Length...");
    colorreset();
    distribute_to_threads(graphs, data, out, ctrl, &general_trend_smodel);
    gather_local_trends(out, ctrl);
    // winning model for each metric
    double best_arhr = 0.0;
    int best_model_arhr = 0;
    int best_walk_step_hr = 0;
    double best_hr = 0.0;
    int best_model_hr = 0;
    int best_walk_step_arhr = 0;
    double best_ndcg = 0.0;
    int best_model_ndcg = 0;
    int best_walk_step_ndcg = 0;
    for (int i = 0; i < data.num_models; i++) {
      for (int j = 0; j < ctrl.max_walk; j++) {
        if (out.general_trends[i][j].HR > best_hr) {
          best_hr = out.general_trends[i][j].HR;
          best_model_hr = i;
          best_walk_step_hr = j;
        }
        if (out.general_trends[i][j].ARHR > best_arhr) {
          best_arhr = out.general_trends[i][j].ARHR;
          best_model_arhr = i;
          best_walk_step_arhr = j;
        }
        if (out.general_trends[i][j].NDCG > best_ndcg) {
          best_ndcg = out.general_trends[i][j].NDCG;
          best_model_ndcg = i;
          best_walk_step_ndcg = j;
        }
      }
    }
    printf("The best models in CV are    : HR:%s ARHR:%s NDCG:%s\n",
           data.item_models[best_model_hr].name,
           data.item_models[best_model_arhr].name,
           data.item_models[best_model_ndcg].name);
    printf("The best scores in CV are    : HR:%f ARHR:%f NDCG:%f\n", best_hr,
           best_arhr, best_ndcg);
    printf("The best walk_steps in CV are: HR:%d ARHR:%d NDCG:%d\n",
           best_walk_step_hr, best_walk_step_arhr, best_walk_step_ndcg);
  } else {
    yellow();
    printf("\nGeneral trends manually set around step %d...\n", ctrl.best_step);
    colorreset();
    set_trends(out, ctrl);
  }

  // exit(0);
  // Save general trends (and possibly values of given step)
  // Also store the name of the item_model
  const char *names[data.num_models];
  for (int i = 0; i < data.num_models; i++)
    names[i] = data.item_models[i].name;

  if (!ctrl.set_trend) // do not save made up trend values
    store_general_trends(out.general_trends, outfiles.outdir_theta,
                         data.num_models, ctrl.max_walk, ctrl.set_trend, names);

  if (ctrl.save_vals)
    store_vals(outfiles.outdir_val, out.val_pred, data.num_models, true);

  // free memory
  csr_array_free(graphs, data.num_models);
}

// Function that trains thetas per user using the cross-validation items
// provided
void perdif_train(data_t data, out_t out, ctrl_t ctrl, output_t outfiles) {

  // Generate model graphs
  csr_graph_t *graphs =
      (csr_graph_t *)malloc(data.num_models * sizeof(csr_graph_t));
  for (int i = 0; i < data.num_models; i++) {
    if (VERBOSE)
      printf("Building the training graphs: %d out of %d \n", i + 1,
             data.num_models);
    graphs[i] = generate_rec_graph(data.users, data.item_models[i],
                                   data.num_users, ctrl.bipartite);
  }

  // if (ctrl.save_vals)
  // 	store_vals(outfiles.outdir_val, out.val_pred, data.num_models, true);

  // Learn personalized diffusion parameters in parralel
  magenta();
  printf("Personalizing the Diffusions...");
  colorreset();
  distribute_to_threads(graphs, data, out, ctrl, &perdif_learn_smodel);

  // computes the inverse mapping to mus
  compute_mus(out.mus, out.thetas, data.num_models);

  // Also store the name of the item_model
  const char *names[data.num_models];
  for (int i = 0; i < data.num_models; i++)
    names[i] = data.item_models[i].name;
  store_parameters(out.thetas, out.mus, outfiles.outdir_theta, data.num_models,
                   names);

  // free memory
  csr_array_free(graphs, data.num_models);
}

void compute_mus(d_mat_t *mus, d_mat_t *thetas, int num_models) {
  //  solve the reccurence that computes the mus
  for (sz_long_t k = 0; k < num_models; k++) {
    mus[k].num_cols = thetas[k].num_cols - 1;
    mus[k].num_rows = thetas[k].num_rows;
    for (sz_long_t i = 0; i < thetas[k].num_rows; i++) {
      double running_sum = 0.0;
      for (sz_long_t j = 0; j < thetas[k].num_cols - 1; j++) {
        double den = 1 - running_sum;
        running_sum += thetas[k].val[i][j];
        double nom = 1 - running_sum;
        mus[k].val[i][thetas[k].num_cols - j - 2] = nom / den;
      }
    }
  }
}

// General-trend generating function for a single model and window of users
void general_trend_smodel(struct pass2thread_t thread_data, int model_counter) {
  // Unpack thread data
  metric_t *local_trend =
      thread_data.out.local_trend[thread_data.u_id][model_counter];
  // csr_graph_t graph = thread_data.graphs[model_counter][thread_data.u_id];
  csr_graph_t graph = thread_data.graphs[model_counter][0];
  user_t *users = thread_data.data.users;
  i_mat_t cv_neg = thread_data.data.cv_neg;
  sz_long_t *cv_pos = thread_data.data.cv_pos;
  ctrl_t ctrl = thread_data.ctrl;
  sz_long_t from_user = thread_data.usr_start;
  sz_long_t num_users = thread_data.usr_win;
  sz_long_t global_num_users = thread_data.data.num_users;
  report_t report = thread_data.report;

  // For storage of tes tvalues
  double p_pos;
  double p_neg[cv_neg.num_rows];

  // Allocate landing prob vectors
  double *p = (double *)malloc(graph.num_nodes * sizeof(double));
  double *p_plus = (double *)malloc(graph.num_nodes * sizeof(double));

  // percentage unit used for progress report
  int one_per_cent = num_users / 100;

  // Iterate over all users
  for (sz_long_t i = from_user; i < from_user + num_users; i++) {
    // Copy user cv ratings to minimize read-conflicts with other threads
    sz_long_t pos = cv_pos[i];
    sz_long_t neg[cv_neg.num_rows];
    for (int j = 0; j < cv_neg.num_rows; j++)
      neg[j] = cv_neg.val[j][i];

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

    // Iterate through every step of the walk
    for (int k = 0; k < ctrl.max_walk; k++) {
      if (k == 0)
        continue;

      // advance one step
      my_CSR_matvec(p_plus, p, graph);
      // Build p_pos and p_neg for user i, step k
      sz_long_t offset = (ctrl.bipartite) ? global_num_users : 0;
      p_pos = p_plus[pos + offset];
      for (int j = 0; j < cv_neg.num_rows; j++)
        p_neg[j] = p_plus[neg[j] + offset];

      // Compute and store user/walk rec metrics
      metric_t metrics = get_rec_metrics(p_pos, p_neg, cv_neg.num_rows);
      local_trend[k].HR += metrics.HR;
      local_trend[k].ARHR += metrics.ARHR;
      local_trend[k].NDCG += metrics.NDCG;

      // Save values for validation purposes
      if (ctrl.save_vals && k == STEP_VAL_SAVE) {
        thread_data.out.val_pred[model_counter].val[i][0] = p_pos;
        for (int j = 0; j < cv_neg.num_rows; j++)
          thread_data.out.val_pred[model_counter].val[i][j + 1] = p_neg[j];
      }

      // Swap pointers
      double *temp = p;
      p = p_plus;
      p_plus = temp;
    }

    // report progress (estimate from one thread)
    if (report.yes && ((i - from_user) % (one_per_cent * report.every) == 0))
      if (one_per_cent > 0) {
        if (i == from_user) { // printf("General trends %d %% complete\n",
                              // report.shift + (int)(i - from_user) /
                              // (one_per_cent * report.every));
          printf("\nProgress: ");
          fflush(stdout);
        } else {
          printf("|");
          fflush(stdout);
        }
      }
    if (report.yes && i == from_user + num_users - 1) {
      printf(" Done!");
      fflush(stdout);
      colorreset();
    }
  }

  // Normalize trends by number of users
  for (int k = 0; k < ctrl.max_walk; k++) {
    local_trend[k].HR /= (double)num_users;
    local_trend[k].ARHR /= (double)num_users;
    local_trend[k].NDCG /= (double)num_users;
  }

  // free
  free(p);
  free(p_plus);
}

// Parameter learning function for a single model and window of users
void perdif_learn_smodel(struct pass2thread_t thread_data, int model_counter) {

  // Unpack thread data

  metric_t *general_trend = thread_data.out.general_trends[model_counter];
  d_mat_t thetas = thread_data.out.thetas[model_counter];
  //	csr_graph_t graph = thread_data.graphs[model_counter][thread_data.u_id];
  csr_graph_t graph = thread_data.graphs[model_counter][0];
  user_t *users = thread_data.data.users;
  i_mat_t cv_neg = thread_data.data.cv_neg;
  sz_long_t *cv_pos = thread_data.data.cv_pos;
  ctrl_t ctrl = thread_data.ctrl;
  sz_long_t from_user = thread_data.usr_start;
  sz_long_t num_users = thread_data.usr_win;
  sz_long_t global_num_users = thread_data.data.num_users;
  report_t report = thread_data.report;

  // Allocate landing prob vectors
  double *p = (double *)malloc(graph.num_nodes * sizeof(double));
  double *p_plus = (double *)malloc(graph.num_nodes * sizeof(double));

  // Square matrix that stores p_values of interest
  d_mat_t P = d_mat_init(cv_neg.num_rows + 1, ctrl.max_walk);

  // For possible coefficients that are used multiple times
  double *coefs = NULL;
  switch (ctrl.which_dif_param) {
  case 0:
    coefs = (double *)malloc(sizeof(double));
    break;
  case 1:
    coefs = (double *)malloc(sizeof(double));
    break;
  case 2:
    coefs = ppr_coefficients(ctrl.max_walk);
    break;
  case 3:
    coefs = hk_coefficients(ctrl.max_walk);
    break;
  case 4:
    coefs = dict_coefficients(ctrl.max_walk);
    break;
  }

  // Pick general trend to regularize according to metric of interest
  // 0) HR, 1)ARHR, 2) NDCG
  double g[ctrl.max_walk];
  double sum_g = 0.0;
  for (int k = 0; k < ctrl.max_walk; k++) {
    if (ctrl.which_target_metric == 0) {
      g[k] = general_trend[k].HR;
    } else if (ctrl.which_target_metric == 1) {
      g[k] = general_trend[k].ARHR;
    } else if (ctrl.which_target_metric == 2) {
      g[k] = general_trend[k].NDCG;
    } else {
      printf("\nERROR: target metric??\n");
    }
    sum_g += g[k];
  }

  // Normalize trend to act as regularizer
  for (int k = 0; k < ctrl.max_walk; k++)
    g[k] /= sum_g;

  // percentage unit used for progress report
  int one_per_cent = num_users / 100;

  // Iterate over all users
  for (sz_long_t i = from_user; i < from_user + num_users; i++) {

    // Copy user cv ratings to minimize read-conflicts with other threads
    sz_long_t pos = cv_pos[i];
    sz_long_t neg[cv_neg.num_rows];
    for (int j = 0; j < cv_neg.num_rows; j++)
      neg[j] = cv_neg.val[j][i];

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
      if (k == 0) {
        // Populate k-th collumn of P
        P.val[0][k] = (ctrl.bipartite) ? p[global_num_users + pos] : p[pos];
        for (int j = 1; j < P.num_rows; j++)
          P.val[j][k] = (ctrl.bipartite) ? p[global_num_users + neg[j - 1]]
                                         : p[neg[j - 1]];
        // sanity check
        // double sum = 0.0;
        // for (int j = 0; j < P.num_rows; j++)
        // 	sum += P.val[j][k];
        // printf("Sum = %f\n",sum);
        continue;
      }
      // Advance one step
      my_CSR_matvec(p_plus, p, graph);

      // Populate k-th collumn of P
      P.val[0][k] =
          (ctrl.bipartite) ? p_plus[global_num_users + pos] : p_plus[pos];
      for (int j = 1; j < P.num_rows; j++)
        P.val[j][k] = (ctrl.bipartite) ? p_plus[global_num_users + neg[j - 1]]
                                       : p_plus[neg[j - 1]];

      // Swap pointers
      double *temp = p;
      p = p_plus;
      p_plus = temp;
    }

    // Call one of the parametr fitting methods from perdif_fit.c
    (*fit_theta[ctrl.which_dif_param])(thetas.val[i], P, g, ctrl.max_walk,
                                       ctrl.lambda, coefs, ctrl.rmse_fit);

    // report progress (estimate from one thread)
    if (report.yes && ((i - from_user) % (one_per_cent * report.every) == 0))
      if (one_per_cent > 0) {
        if (i == from_user) {
          // printf("Learning parameters %d %% complete\n", report.shift +
          // (int)(i - from_user) / (one_per_cent * report.every));
          printf("\nProgress: ");
          fflush(stdout);
        } else {
          printf("|");
          fflush(stdout);
        }
      }
    if (report.yes && i == from_user + num_users - 1) {
      printf(" Done!");
      fflush(stdout);
      colorreset();
    }
  }

  // free
  free(coefs);
  free(p);
  free(p_plus);
  d_mat_free(P);
}

// Aggregate all local threads to get general trends
static void gather_local_trends(out_t out, ctrl_t ctrl) {
  for (int j = 0; j < out.num_models; j++) {
    for (int i = 0; i < ctrl.usr_threads; i++) {
      // Local trends weighted in case they contain different ammount of users
      int usr_win = out.num_users / ctrl.usr_threads;
      if (i == ctrl.usr_threads - 1)
        usr_win = out.num_users - (ctrl.usr_threads - 1) * usr_win;

      double weight = (double)usr_win / (double)out.num_users;

      for (int k = 0; k < ctrl.max_walk; k++) {
        out.general_trends[j][k].HR += weight * out.local_trend[i][j][k].HR;
        out.general_trends[j][k].ARHR += weight * out.local_trend[i][j][k].ARHR;
        out.general_trends[j][k].NDCG += weight * out.local_trend[i][j][k].NDCG;
      }
    }
  }
}

// Build "general trends" linearly around preselected best-step
static void set_trends(out_t out, ctrl_t ctrl) {

  if (ctrl.best_step > ctrl.max_walk) {
    printf("ERROR: Best step must be <= K_max\n");
    exit(EXIT_FAILURE);
  }

  double trend[ctrl.max_walk];

  double base = REG_BASE;

  for (int i = 0; i < ctrl.max_walk; i++)
    trend[i] = pow(base, -1.0 * abs((double)(i - ctrl.best_step)));

  for (int j = 0; j < out.num_models; j++) {
    for (int i = 0; i < ctrl.max_walk; i++) {
      out.general_trends[j][i].HR = trend[i];
      out.general_trends[j][i].ARHR = trend[i];
      out.general_trends[j][i].NDCG = trend[i];
    }
  }
}