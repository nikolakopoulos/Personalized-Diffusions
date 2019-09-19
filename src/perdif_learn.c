///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains the PERDIF program that implements the Personalized Diffusions
 Recommendations framework.

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
#include <mkl.h>

#include "perdif_predict.h"
#include "perdif_train.h"
#include "rec_IO.h"
#include "rec_defs.h"
#include "rec_mem.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN
int main(int argc, char **argv) {
  printf("\n---------------------------------------------------------------\n");
  printf("Personalized Diffusions, version %s\n", PERDIF_VERSION);
  printf("---------------------------------------------------------------\n");
  
  // Parse command-line arguments
  cmd_args_t args;
  parse_commandline_args(argc, argv, &args);

  // Load data
  data_t data = load_data(args.input, &args.ctrl);
  if (data.no_models)
    printf("Error: No item models were found..\n\n");

  // Workspace to store thetas and predictions
  out_t output = output_init(data, args.ctrl);

  // Train parameters
  struct timespec start, finish;
  double elapsed;
  mkl_set_threading_layer(MKL_THREADING_GNU); 
  clock_gettime(CLOCK_MONOTONIC, &start);
  perdif_train(data, output, args.ctrl, args.output);
  clock_gettime(CLOCK_MONOTONIC, &finish);

  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

  printf("Time to learn diffusions for all users: %f sec\n", elapsed);

  // Predict items by applying thetas
  perdif_predict(data, output, args.ctrl, args.output);

  // Free
  data_free(data);
  output_free(output, args.ctrl);
  return 0;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
