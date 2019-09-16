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
#include "rec_IO.h"
#include "rec_defs.h"
#include "rec_graph_gen.h"
#include "rec_mem.h"
#include "rec_metrics.h"

static void *single_thread_dist(void *);

// distributes computations to multiple threads
void distribute_to_threads(csr_graph_t *graphs, data_t data, out_t out,
                           ctrl_t ctrl,
                           void (*func)(struct pass2thread_t, int)) {

  // Prepare data to be passed to each thread
  // model threads cannot be more than the actual number of models to be
  // processes
  int model_threads = (ctrl.model_threads >= data.num_models)
                          ? data.num_models
                          : ctrl.model_threads;

  int model_win = data.num_models / model_threads;
  int usr_win = data.num_users / ctrl.usr_threads;
  int last_model_win = data.num_models - (model_threads - 1) * model_win;
  int last_usr_win = data.num_users - (ctrl.usr_threads - 1) * usr_win;

  // Copy graphs to be istributed to different usr threads
  csr_graph_t **graph_copies =
      (csr_graph_t **)malloc(usr_win * sizeof(csr_graph_t *));
  for (int i = 0; i < data.num_models; i++)
    graph_copies[i] = &graphs[i];

  // Prepare thread parameters
  struct pass2thread_t param_1[ctrl.usr_threads][model_threads];

  for (int i = 0; i < ctrl.usr_threads; i++) {
    for (int j = 0; j < model_threads; j++) {
      param_1[i][j] =
          (struct pass2thread_t){.func = func,
                                 .m_id = j,
                                 .u_id = i,
                                 .model_start = j * model_win,
                                 .model_win = model_win,
                                 .usr_start = i * usr_win,
                                 .usr_win = usr_win,
                                 .graphs = graph_copies,
                                 .data = data,
                                 .out = out,
                                 .ctrl = ctrl,
                                 .report = (report_t){.yes = false}};
    }
    param_1[i][model_threads - 1].model_win = last_model_win;
  }
  for (int j = 0; j < model_threads; j++)
    param_1[ctrl.usr_threads - 1][j].usr_win = last_usr_win;

  // Only last (bottom right corner) thread reports progress
  param_1[ctrl.usr_threads - 1][model_threads - 1].report =
      (report_t){.yes = true, .every = last_model_win};

  // Spawn threads and start running
  pthread_t tid[ctrl.usr_threads][model_threads];
  for (int i = 0; i < ctrl.usr_threads; i++) {
    for (int j = 0; j < model_threads; j++) {
      pthread_create(&tid[i][j], NULL, single_thread_dist,
                     (void *)(*(param_1 + i) + j));
    }
  }

  // Wait for all threads to finish before continuing
  for (int i = 0; i < ctrl.usr_threads; i++) {
    for (int j = 0; j < model_threads; j++) {
      pthread_join(tid[i][j], NULL);
    }
  }
  printf("\n");

  // free graph copies and local trends
  // for (int i = 0; i < data.num_models; i++)
  // csr_array_free(graph_copies[i], ctrl.usr_threads);
  free(graph_copies);
}

// Distributes models to single thread
static void *single_thread_dist(void *param) {

  struct pass2thread_t thread_data = *(struct pass2thread_t *)param;

  if (VERBOSE)
    printf("Thread # (%d , %d) started with %d models and %d users..\n",
           thread_data.m_id, thread_data.u_id, thread_data.model_win,
           (int)thread_data.usr_win);

  for (int model_counter = thread_data.model_start;
       model_counter < thread_data.model_start + thread_data.model_win;
       model_counter++) {
    if (thread_data.report.yes)
      thread_data.report.shift = (100 / thread_data.model_win) *
                                 (model_counter - thread_data.model_start);

    thread_data.func(thread_data, model_counter);
  }

  if (VERBOSE)
    printf("Thread # (%d , %d) finished..\n", thread_data.m_id,
           thread_data.u_id);

  pthread_exit(0);
}