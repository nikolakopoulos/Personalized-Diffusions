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

#include "rec_defs.h"
#include "rec_metrics.h"

// Returns HR, ARHR and NDCG metrics
metric_t get_rec_metrics(double p_pos, double *p_neg, int num_neg) {

  metric_t metric = (metric_t){.HR = 0.0, .ARHR = 0.0, .NDCG = 0.0};

  int position = 0;

  for (int i = 0; i < num_neg; i++) {
    position = (p_neg[i] >= p_pos) ? position + 1 : position;
  }
  position++;

  // If NOT in TOP_N return zero for all metrics
  if (position <= TOP_N) {
    metric.HR = 1.0;
    metric.ARHR = 1.0 / (double)position;
    metric.NDCG = 1.0 / log2((double)(1 + position));
  }

  return metric;
}
