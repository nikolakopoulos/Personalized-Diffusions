////////////////////////////////////////////////////////////////////////////////////////////////

/*
 Contains high-level routines for parsing the command line for arguments

 Code by: Dimitris Berberidis and Athanasios N. Nikolakopoulos
 University of Minnesota 2018-2019
*/

///////////////////////////////////////////////////////////////////////////////////////////////

#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "csr_handling.h"
#include "rec_IO.h"
#include "rec_defs.h"
#include "rec_mem.h"

// easier handling of the item models
#include <dirent.h>
#include <stdarg.h>

/* Arrange the N elements of ARRAY in random order.
   Only effective if N is much smaller than RAND_MAX;
   if this may not be the case, use a better random
   number generator. */
void shuffle(int *array, size_t n) {
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

char *concat_thanos(int count, ...) {
  va_list ap;
  int i;

  // Find required length to store merged string
  size_t len = 1; // room for NULL
  va_start(ap, count);
  for (i = 0; i < count; i++)
    len += strlen(va_arg(ap, char *));
  va_end(ap);

  // Allocate memory to concat strings
  char *merged = calloc(sizeof(char), len);
  int null_pos = 0;

  // Actually concatenate strings
  va_start(ap, count);
  for (i = 0; i < count; i++) {
    char *s = va_arg(ap, char *);
    strcpy(merged + null_pos, s);
    null_pos += strlen(s);
  }
  va_end(ap);

  return merged;
}

static void load_v(char *, sz_long_t **, i_mat_t *);
static void reverse(char *);
static void itoa(int, char *);
static sz_long_t *csr_line_to_ind(char *, sz_med_t *, bool);
static sz_med_t csr_read_line(char *, long double *, sz_long_t *);

// List of diffusion parametrizations ( MUST be aligned with.. )
static const char *dif_param_list[] = {"free", "single-best", "ppr", "hk",
                                       "dictionary"};

// List of target metrics ( MUST be aligned with.. )
static const char *metrics_list[] = {"HR", "ARHR", "NDCG"};

// Parsing command line arguments with getopt_long_only
void parse_commandline_args(int argc, char **argv, cmd_args_t *args) {

  // set default arguments
  (*args) = (cmd_args_t){.ctrl.max_walk = DEFAULT_MAX_WALK,
                         .input.rating_mat = DEFAULT_RATING_MAT,
                         .input.rating_mat_full = DEFAULT_RATING_MAT_FULL,
                         .input.item_models_dir = DEFAULT_ITEM_MODELS_DIR,
                         .input.CV_item_models_dir = DEFAULT_CV_ITEM_MODELS_DIR,
                        //  .ctrl.usr_threads = omp_get_max_threads()/2,
                         .ctrl.usr_threads = 20,
                         .ctrl.model_threads = DEFAULT_MODEL_THREADS,
                         .ctrl.num_threads = DEFAULT_NUM_THREADS,
                         .output.outdir_pred = DEFAULT_OUTDIR_PRED,
                         .output.outdir_theta = DEFAULT_OUTDIR_THETA,
                         .output.outdir_val = DEFAULT_OUTDIR_VAL,
                         .ctrl.which_dif_param = DEFAULT_DIF_PARAM,
                         .ctrl.bipartite = DEFAULT_BIPARTITE,
                         .ctrl.rmse_fit = DEFAULT_RMSE_FIT,
                         .ctrl.itm_trsp = DEFAULT_ITM_TRSP,
                         .ctrl.save_vals = DEFAULT_SAVE_VALS,
                         .ctrl.best_step = DEFAULT_BEST_STEP,
                         .input.val_mat = DEFAULT_VAL_MAT,
                         .input.test_mat = DEFAULT_TEST_MAT,
                         .ctrl.which_target_metric = DEFAULT_TARGET_METRIC};

  int opt = 0;
  // Specifying the expected options
  // The two options l and b expect numbers as argument
  static struct option long_options[] = {
      {"max_walk", required_argument, 0, 'd'},
      {"usr_threads", required_argument, 0, 'e'},
      {"strategy", required_argument, 0, 'j'},
      {"bpr_fit", no_argument, 0, 'm'},
      {"target_metric", required_argument, 0, 'o'},
      {"itm_trsp", no_argument, 0, 'y'},
      {"save_vals", no_argument, 0, 'x'},
      {"num_threads", required_argument, 0, 'l'},
      {"dataset", required_argument, 0, 'q'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  static char helpstr[][512] = {
      " ",
      " Usage:",
      "   perdif_learn [options]",
      " ",
      " Options:",
      " ",
      "   -dataset=string",
      "      Specifies the dataset to be used.",
      "        The dataset name is assumed to correspond to the name of the dataset folder in data/in and data/out directories.",
      "        The default value is ml1m",
      " ",
      "   -max_walk=int",
      "      Specifies that length of the personalized item exploration walks.",
      "      The default value is 5",
      " ",
      "   -strategy=string",
      "      Available options are:",      
      "        single-best  -  Chooses for each user the Kth step that minimizes training error [default].",
      "        free         -  The PerDIF^Free model.",
      "        dictionary   -  The PerDIF^Par model.",
      "        hk           -  PerDIF^par using only Heat Kernel weights.",
      "        ppr          -  PerDIF^par using only Personalized PageRank weights.",
      " ",
      "   -bpr_fit",
      "      It fits the personalized diffusions using a BRP loss. Default is RMSE",
      " ",
      "   -usr_threads=int",
      "      Specifies the number of threads to be used for learning and evaluating the model.",
      "      The default value is maximum number of threads available on the machine.",
      " ",
      "   -help",
      "      Prints this message.",
      " ",
      " Example run: ./perdif_learn -dataset=ml1m -max_walk=6 -strategy=dictionary",
      " ",
      ""};

  int long_index = 0;
  bool method_found = false, metric_found = false;
  // omp_get_max_threads();
  while ((opt = getopt_long_only(argc, argv, "", long_options, &long_index)) !=
         -1) {
    switch (opt) {
    case 'd':
      args->ctrl.max_walk =
          atoi(optarg) + 1; // the +1 is for using the max_walk as a barrier.
      // args->ctrl.max_walk = atoi(optarg);
      if (args->ctrl.max_walk < 1) {
        printf("ERROR: Max length of walks must be >=1\n");
        exit(EXIT_FAILURE);
      }
      break;
    case 'e':
      args->ctrl.usr_threads = atoi(optarg);
      if (args->ctrl.usr_threads < 1) {
        printf("ERROR: Number of user threads must be >=1\n");
        exit(EXIT_FAILURE);
      }
      break;
    case 'j':
      for (sz_short_t i = 0; i < NUM_METHODS; i++) {
        if (!strcmp(optarg, dif_param_list[i])) {
          args->ctrl.which_dif_param = i;
          method_found = true;
          if(i>1){ // less usr_threads are usually better here because more time is being spend in the optimization problem
            args->ctrl.usr_threads = 1;
            printf("Setting usr_threads to 1\n");
          }
        }
      }
      if (!method_found) {
        printf("ERROR: Parametrization type not recognized\n");
        exit(EXIT_FAILURE);
      }
      break;
    case 'y':
      args->ctrl.itm_trsp = false;
      break;
    case 'm':
      args->ctrl.rmse_fit = false;
      break;
    case 'x':
      args->ctrl.save_vals = true;
      break;
    case 'o':
      for (sz_short_t i = 0; i < NUM_METRICS; i++) {
        if (!strcmp(optarg, metrics_list[i]))
          args->ctrl.which_target_metric = i;
        metric_found = true;
      }
      if (!metric_found)
        printf("ERROR: Metric type not recognized\n");
      break;
    case 'l':
      args->ctrl.num_threads = atoi(optarg);
      if (args->ctrl.num_threads < 1) {
        printf("ERROR: Number of threads must be >=1\n");
        exit(EXIT_FAILURE);
      }
      break;
    case 'q':
      // Input
      args->input.rating_mat =
          concat_thanos(3, DEFAULT_IN_DIR, optarg, "/R.csr");
      args->input.rating_mat_full =
          concat_thanos(3, DEFAULT_IN_DIR, optarg, "/R_full.csr");
      args->input.item_models_dir =
          concat_thanos(3, DEFAULT_IN_DIR, optarg, "/selected_item_models");
      args->input.CV_item_models_dir =
          concat_thanos(3, DEFAULT_IN_DIR, optarg, "/CV_item_models");
      // Val
      args->input.val_mat = concat_thanos(3, DEFAULT_IN_DIR, optarg, "/CV.csr");
      // Test
      args->input.test_mat =
          concat_thanos(3, DEFAULT_OUT_DIR, optarg, "/TestSet.csr");
      // Output
      args->output.outdir_pred =
          concat_thanos(3, DEFAULT_OUT_DIR, optarg, "/predictions");
      args->output.outdir_theta =
          concat_thanos(3, DEFAULT_OUT_DIR, optarg, "/model_parameters");
      args->output.outdir_val =
          concat_thanos(3, DEFAULT_OUT_DIR, optarg, "/val_workspace");
      break;
    case 'h':
      for (int i = 0; strlen(helpstr[i]) > 0; i++)
        printf("%s\n", helpstr[i]);
      exit(0);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

// Load data from input files
data_t load_data(input_t input, ctrl_t *ctrl) {

  data_t data = data_init();

  // Read rating matrices
  //
  blue();
  printf("Loading Data...");
  colorreset();
  printf("\nImplicit Feedback Matrix (training)...");
  data.users = read_rating_matrix_csr(&data.num_users, input.rating_mat);
  printf("Done!\n");
  printf("Full Implicit Feedback Matrix...");
  sz_long_t test_dimension;
  data.users_full =
      read_rating_matrix_csr(&test_dimension, input.rating_mat_full);
  printf("Done!\n");
  if (test_dimension != data.num_users) {
    red();
    printf("ERROR: Training and full matrix dimension mismatch..\n");
    colorreset();
  }

  // Read item graphs from file
  printf("Item model(s)...");
  data.num_models = DEFAULT_NUM_MODELS;
  data.item_models = read_item_models(input.item_models_dir, &data.num_models);
  printf("%d item model(s) found...", data.num_models);
  printf("Done!\n");
  if (VERBOSE)
    printf("%d users.. \n", (int)data.num_users);
  // Automatically adjust model/user threads if only total num_threads is given
  // Requires knowledge of number of models
  if (ctrl->num_threads) {
    if (ctrl->num_threads <= data.num_models) {
      ctrl->usr_threads = 1;
      ctrl->model_threads = ctrl->num_threads;
    } else {
      ctrl->model_threads = data.num_models;
      ctrl->usr_threads = (ctrl->num_threads / data.num_models);
    }
  }

  // Read cross validation and test items
  printf("Validation-Set...");
  fflush(stdout);
  load_v(input.val_mat, &data.cv_pos, &data.cv_neg);
  printf("Done!\n");
  printf("Test-Set...");
  fflush(stdout);
  load_v(input.test_mat, &data.test_pos, &data.test_neg);
  printf("Done!\n");

  if (data.test_neg.num_rows != data.cv_neg.num_rows ||
      data.test_neg.num_cols != data.cv_neg.num_cols) {
    red();
    printf("ERROR: Test and validation matrix dimension mismatch!\n");
    colorreset();
  }

  if (data.num_models == 0) {
    usr_free(data.users, data.num_users);
    data.no_models = true;
  }

  return data;
}

// Load data from input files
data_t load_CV_data(input_t input, ctrl_t *ctrl) {

  data_t data = datacv_init();

  // Read rating matrices
  //
  blue();
  printf("Loading Data...");
  colorreset();
  printf("\nImplicit Feedback Matrix (training)...");
  data.users = read_rating_matrix_csr(&data.num_users, input.rating_mat);
  printf("Done!\n");

  // Read item graphs from file
  printf("Item model(s)...");
  data.num_models = DEFAULT_NUM_MODELS;
  data.item_models =
      read_item_models(input.CV_item_models_dir, &data.num_models);
  printf("%d item model(s) found...", data.num_models);
  printf("Done!\n");
  if (VERBOSE)
    printf("%d users.. \n", (int)data.num_users);
  // Automatically adjust model/user threads if only total num_threads is given
  // Requires knowledge of number of models
  if (ctrl->num_threads) {
    if (ctrl->num_threads <= data.num_models) {
      ctrl->usr_threads = 1;
      ctrl->model_threads = ctrl->num_threads;
    } else {
      ctrl->model_threads = data.num_models;
      ctrl->usr_threads = (ctrl->num_threads / data.num_models);
    }
  }

  // Read cross validation and test items
  printf("Validation-Set...");
  fflush(stdout);
  load_v(input.val_mat, &data.cv_pos, &data.cv_neg);
  printf("Done!\n");

  if (data.num_models == 0) {
    usr_free(data.users, data.num_users);
    data.no_models = true;
  }

  return data;
}

// Read matrix with cross validation or test ratings
static void load_v(char *filename, sz_long_t **v_pos, i_mat_t *v_neg) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    red();
    printf("ERROR: Cannot open rating matrix file\n");
    colorreset();
  }
  int len = 2 * USR_BUFF_SIZE * sizeof(sz_long_t);
  char *line = (char *)malloc(len * sizeof(char));
  sz_long_t count_rows = 0;
  sz_med_t count_cols = 0;

  while (fgets(line, len, file) != NULL) {
    sz_long_t *v_inds = csr_line_to_ind(line, &count_cols, true);

    if (count_rows == 0) {
      memcpy(*v_pos, v_inds, count_cols * sizeof(sz_long_t));
      *v_pos = realloc(*v_pos, count_cols * sizeof(sz_long_t));
    } else {
      memcpy(v_neg->val[count_rows - 1], v_inds,
             count_cols * sizeof(sz_long_t));
    }
    free(v_inds);
    count_rows++;
  }
  fclose(file);

  free(line);
  i_mat_resize(v_neg, count_rows - 1, count_cols);
}


// Reads feedback matrix in csr format from file and converts it to an array of
// users Also returns the total number of users
user_t *read_rating_matrix_csr(sz_long_t *usr_count, char *filename) {

  user_t *users = (user_t *)malloc(USR_BUFF_SIZE * sizeof(user_t));

  FILE *file = fopen(filename, "r");
  if (!file) {
    red();
    printf("ERROR: Cannot open rating matrix file\n");
    colorreset();
  }

  int len = 2 * ITM_BUFF_SIZE * sizeof(sz_long_t);
  char line[len];
  *usr_count = 0;

  while (fgets(line, len, file) != NULL) {
    users[*usr_count].items =
        csr_line_to_ind(line, &users[*usr_count].num_items, false);
    if (!users[*usr_count].num_items)
      printf("WARNING: User %d has 0 ratings..\n", (int)*usr_count);

    *usr_count += 1;
  }
  fclose(file);

  user_t *temp = realloc(users, (*usr_count) * sizeof(user_t));

  if (temp == NULL) {
    red();
    printf("ERROR REALLOCATING\n");
    colorreset();
  } else {
    users = temp;
  }

  return users;
}

// Reads item_graph in csr form from file to csr graphs
csr_graph_t read_item_graph_csr(char *filename, char *modelname) {

  csr_graph_t item_graph;

  csr_alloc(&item_graph);

  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    red();
    printf("ERROR: Cannot open item model file");
    colorreset();
  }

  // store the name of the model
  strcpy(item_graph.name, modelname);

  int len = 2 * ITM_BUFF_SIZE * sizeof(sz_long_t);
  char line[len];
  item_graph.num_nodes = 0;
  item_graph.nnz = 0;

  item_graph.csr_row_pointer[0] = 0;
  while (fgets(line, len, file) != NULL) {
    sz_med_t num_item_edges =
        csr_read_line(line, &item_graph.csr_value[item_graph.nnz],
                      &item_graph.csr_column[item_graph.nnz]);
    item_graph.csr_row_pointer[item_graph.num_nodes + 1] =
        item_graph.csr_row_pointer[item_graph.num_nodes] + num_item_edges;
    item_graph.nnz += num_item_edges;
    item_graph.num_nodes++;
  }
  fclose(file);

#if DEBUG
  printf("nnz: %" PRIu64 "\n", item_graph.nnz);
#endif

  csr_realloc(&item_graph, item_graph.nnz, item_graph.num_nodes);

  return item_graph;
}

// Reads all item models in a directory
// The Item-model files in the directory must end in ".model"
csr_graph_t *read_item_models(char *dirname, int *num_models) {
  *num_models = 0;
  csr_graph_t *models =
      (csr_graph_t *)malloc(MAX_ITEM_MODELS * sizeof(csr_graph_t));

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(dirname)) != NULL) {
    int i = 0;
    while ((ent = readdir(dir)) != NULL) {
      if (strstr(ent->d_name, ".model") !=
          NULL) // check if the listed file contains the ".model" substring
      {
        char filename[200]; // read the model into an array
        strcpy(filename, dirname);
        strcat(filename, "/");
        strcat(filename, ent->d_name);
        models[i] = read_item_graph_csr(filename, ent->d_name);
        *num_models += 1;
        i++;
      }
    }
    closedir(dir);
  } else {
    /* could not open directory */
    perror("");
    exit(EXIT_FAILURE);
  }

  models = realloc(models, (*num_models) * sizeof(csr_graph_t));
  return models;
}

// Convert string to integer array
// Numbers are delimited by space and every second number is ignored (is always
// 1)
// If vals=true it loads the values of non zero entries
// If vals = false it loads the indices
static sz_long_t *csr_line_to_ind(char *line, sz_med_t *num_entries,
                                  bool vals) {

  sz_long_t *items = (sz_long_t *)malloc(ITM_BUFF_SIZE * sizeof(sz_long_t));
  sz_med_t temp;
  char seps[3] = " ,\t";
  *num_entries = 0;
  sz_med_t count = 0;
  char *token = strtok(line, seps);
  while (token != NULL) {
    sscanf(token, "%" SCNu32, &temp);

    bool load;
    if (vals) {
      load = (count % 2 == 1);
      count++;
    } else {
      load = !(count++ % 2);
    }

    if (load) {
      items[*num_entries] = (sz_long_t)(temp - 1);
      *num_entries += 1;
    }

    token = strtok(NULL, seps);
  }

  if (!vals)
    *num_entries -= 1; // Because the last is just the newline character

  items = (sz_long_t *)realloc(items, *num_entries * sizeof(sz_long_t));

  return items;
}

// Break string to csr matrix line (values and collumns)
// Newline character is also written on collumn but it is ignored
static sz_med_t csr_read_line(char *line, long double *value,
                              sz_long_t *column) {

  char seps[3] = " ,\t";
  sz_med_t num_val = 0, num_col = 0, num_tokens = 0;

  char *token = strtok(line, seps);
  while (token != NULL) {
    if (token[0] == '\n')
      break; // End of tokens, go to next line

    if (!(num_tokens % 2)) {
      sscanf(token, "%" SCNu64, &column[num_col++]);
      column[num_col - 1] -= 1;
    } else {
      sscanf(token, "%Lf", &value[num_val++]);
    }

    num_tokens++;

    token = strtok(NULL, seps);
  }

  if (num_val != num_col && num_tokens > 1) {
    red();
    printf("ERROR while reading item graph\n");
    colorreset();
  }

  return num_val;
}

// Writes double matrix to file
void write_d_mat(d_mat_t mat, char *filename) {
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    red();
    printf("Failure to open outfile.\n");
    colorreset();
  }
  for (sz_long_t i = 0; i < mat.num_rows; i++) {
    for (sz_long_t j = 0; j < mat.num_cols; j++)
      fprintf(file, "%0.10f ", mat.val[i][j]);
    fprintf(file, "\n");
  }
  fclose(file);
}

/* itoa:  convert n to characters in s */
static void itoa(int n, char *s) {
  int i, sign;

  if ((sign = n) < 0) /* record sign */
    n = -n;           /* make n positive */
  i = 0;
  do {                     /* generate digits in reverse order */
    s[i++] = n % 10 + '0'; /* get next digit */
  } while ((n /= 10) > 0); /* delete it */
  if (sign < 0)
    s[i++] = '-';
  s[i] = '\0';
  reverse(s);
}

/* reverse:  reverse string s in place */
static void reverse(char *s) {
  int i, j;
  char c;

  for (i = 0, j = strlen(s) - 1; i < j; i++, j--) {
    c = s[i];
    s[i] = s[j];
    s[j] = c;
  }
}

// Store the general trends for different models
void store_general_trends(metric_t **general_trends, char *outdir,
                          int num_models, int max_walk, const char **names) {

  if (VERBOSE) {
    printf("\nSaving general trends.. \n\n");
    printf("\n%d files in %s.. \n\n", num_models, outdir);
  }

  for (int i = 0; i < num_models; i++) {
    char *filename = concat_thanos(4, outdir, "/", names[i], ".trend");

    // check if file exists
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
      red();
      printf("\nERROR: Failed to open %d-th trend file\n", i);
      colorreset();
    }

    fprintf(file, "#HR\tARHR\tNDCG\n");
    for (int k = 0; k < max_walk; k++)
      fprintf(file, "%0.4f\t%0.4f\t%0.4f\n", general_trends[i][k].HR,
              general_trends[i][k].ARHR, general_trends[i][k].NDCG);

    fclose(file);

    if (VERBOSE)
      printf("%s\n", filename);
  }
  printf("\n");
}

// Store validation or test rating predicted values to given directory

void store_vals(char *outdir, d_mat_t *mat, int num_models, bool validation) {
  if (VERBOSE) {
    if (validation) {
      printf("\nSaving validation values of step %d  \n", (int)STEP_VAL_SAVE);
    } else {
      printf("\nSaving test values\n");
    }
    printf("\n%d files in %s \n\n", num_models, outdir);
  }

  for (int i = 0; i < num_models; i++) {
    char filename[200], buffer[5], buffer2[5];
    strcpy(filename, outdir);
    strcat(filename, "/w");
    itoa(i, buffer);
    strcat(filename, buffer);
    if (validation) {
      strcat(filename, "_step_");
      itoa((int)(STEP_VAL_SAVE + 1), buffer2);
      strcat(filename, buffer2);
    }
    strcat(filename, ".pred");

    write_d_mat(mat[i], filename);
  }
}

// store per user diffusion parameters
void store_parameters(d_mat_t *thetas, d_mat_t *mus, char *outdir,
                      int num_models, const char **names) {
  if (VERBOSE) {
    printf("\nSaving Personalized Diffusion Parameters... \n\n");
    printf("\n%d files in %s.. \n\n", num_models, outdir);
  }

  for (int i = 0; i < num_models; i++) {
    char *filename;
    filename = concat_thanos(4, outdir, "/", names[i], ".theta");
    write_d_mat(thetas[i], filename);
    if (VERBOSE)
      printf("%s\n", filename);

    char *filename2;
    filename2 = concat_thanos(4, outdir, "/", names[i], ".mu");
    write_d_mat(mus[i], filename2);
    if (VERBOSE)
      printf("%s\n", filename2);
  }
}

// store and print test results
void store_results(metric_t **result, char *outdir, int num_models,
                   sz_long_t num_users, const char **names) {

  if (VERBOSE) {
    printf("\nSaving per user test results in %s.. \n\n", outdir);
  }

  cyan();
  printf("\nRESULTS : ");

  for (int j = 0; j < num_models; j++) {
    char *filename = concat_thanos(4, outdir, "/", names[j], ".results");

    double HR_mean = 0.0, ARHR_mean = 0.0, NDCG_mean = 0.0;

    FILE *file = fopen(filename, "w");
    fprintf(file, "#user\tHR\tARHR\tNDCG\n");
    fprintf(file, "\n");
    for (sz_long_t i = 0; i < num_users; i++) {
      fprintf(file, "%d\t%0.4f\t%0.4f\t%0.4f\n", (int)i, result[j][i].HR,
              result[j][i].ARHR, result[j][i].NDCG);
      HR_mean += result[j][i].HR;
      ARHR_mean += result[j][i].ARHR;
      NDCG_mean += result[j][i].NDCG;
    }
    HR_mean /= (double)num_users;
    ARHR_mean /= (double)num_users;
    NDCG_mean /= (double)num_users;
    fprintf(file, "\n");
    fprintf(file, "MEAN:\t%0.4f\t%0.4f\t%0.4f\n", HR_mean, ARHR_mean,
            NDCG_mean);
    fclose(file);
    if (j > 0)
      printf("          %s: HR=%0.4f  ARHR=%0.4f  NDCG=%0.4f\n", names[j],
             HR_mean, ARHR_mean, NDCG_mean);
    else
      printf("%s: HR=%0.4f  ARHR=%0.4f  NDCG=%0.4f\n", names[j], HR_mean,
             ARHR_mean, NDCG_mean);
  }
  colorreset();
  printf("\n");
}

void green() { printf("\x1b[32m"); }

void red() { printf("\033[1;31m"); }

void yellow() { printf("\033[1;33m"); }

void magenta() { printf("\x1b[35m"); }

void blue() { printf("\x1b[34m"); }

void cyan() { printf("\x1b[36m"); }

void colorreset() { printf("\033[0m"); }