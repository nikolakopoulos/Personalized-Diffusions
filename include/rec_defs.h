#ifndef REC_DEFS_H_
#define REC_DEFS_H_

#include <stdbool.h>

#define DEBUG false
#define PRINT_THETAS false

// PerDIF version
#define PERDIF_VERSION "1.0"

// Input buffer sizes

#define USR_BUFF_SIZE 500000
#define ITM_BUFF_SIZE 400000

// Default command line arguments

#define DEFAULT_MAX_WALK 10

#define DEFAULT_DATASET "ml1m"

#define DEFAULT_IN_DIR "./data/in/"

#define DEFAULT_OUT_DIR "./data/out/"

#define DEFAULT_RATING_MAT DEFAULT_IN_DIR DEFAULT_DATASET "/R.csr"

#define DEFAULT_RATING_MAT_FULL DEFAULT_IN_DIR DEFAULT_DATASET "/R_full.csr"

#define DEFAULT_ITEM_MODELS_DIR DEFAULT_IN_DIR DEFAULT_DATASET "/selected_item_models"

#define DEFAULT_CV_ITEM_MODELS_DIR DEFAULT_IN_DIR DEFAULT_DATASET "/CV_item_models"

#define DEFAULT_NUM_MODELS 1

#define DEFAULT_USR_THREADS 1

#define DEFAULT_MODEL_THREADS 1

#define DEFAULT_NUM_THREADS 0

#define VERBOSE false

#define DEFAULT_OUTDIR_PRED DEFAULT_OUT_DIR DEFAULT_DATASET "/predictions"

#define DEFAULT_OUTDIR_THETA DEFAULT_OUT_DIR DEFAULT_DATASET "/model_parameters"

#define DEFAULT_OUTDIR_VAL DEFAULT_OUT_DIR DEFAULT_DATASET "/val_workspace"

#define DEFAULT_DIF_PARAM 1 // single best

#define DEFAULT_BIPARTITE false

#define DEFAULT_BEST_STEP 0

#define DEFAULT_SET_TREND false

// #define DEFAULT_LAMBDA 1.0
#define DEFAULT_LAMBDA 0.0

#define DEFAULT_ITM_TRSP true

#define DEFAULT_RMSE_FIT true

#define DEFAULT_SAVE_VALS false

#define DEFAULT_VAL_MAT DEFAULT_IN_DIR DEFAULT_DATASET "/CV.csr"

#define DEFAULT_TEST_MAT DEFAULT_OUT_DIR DEFAULT_DATASET "/TestSet.csr"

#define DEFAULT_TARGET_METRIC 0 // HR

// Default optimization parameters

#define REG_BASE 1.2

#define GRID_POINTS 15

#define MAX_HK_COEF 10.0

#define MAX_ITEM_MODELS 1000

#define NUM_METHODS 5

#define NUM_METRICS 3

#define NUM_UNSEEN 1000

#define TOP_N 10

#define DEFAULT_SUBSET_SIZE 1000

#define ALPHA 0.005

#define GD_TOL 1.0e-6

#define GD_TOL_2 1.0e-6

#define STEPSIZE 0.1

#define STEPSIZE_2 0.95

#define MAXIT_GD 10000

#define PROJ_TOL 1.0e-4

#define L2_REG_LAMBDA 0.0

#define STEP_VAL_SAVE 6

// INTEGER TYPES

typedef uint64_t
    sz_long_t; // Long range unsinged integer. Used for node and edge indexing.

typedef uint32_t sz_med_t; // Medium range unsinged integer. Used for random
                           // walk length, iteration indexing and seed indexing

typedef uint8_t sz_short_t; // Short range unsigned integer. Used for class and
                            // thread indexing.

typedef int8_t class_t; // Short integer for actual label values.

// DATA STRUCTURES

// Record of user-item matrix as an array of user STRUCTURES

typedef struct {
  sz_long_t *items;
  sz_med_t num_items;
} user_t;

// Record of item-user matrix as an array of item STRUCTURES

typedef struct {
  sz_long_t *users;
  sz_med_t num_users;
} item_t;

// Double and index struct for sorting and keeping indexes

typedef struct {
  double val;
  int ind;
} val_and_ind_t;

// Input files/directories
typedef struct {
  char *rating_mat;
  char *rating_mat_full;
  char *item_models_dir;
  char *CV_item_models_dir;
  char *val_mat;
  char *test_mat;
} input_t;

// Output files/directories
typedef struct {
  char *outdir_pred;
  char *outdir_theta;
  char *outdir_val;
} output_t;

// Algorithm parameters
typedef struct {
  int max_walk;
  int usr_threads;
  int model_threads;
  int num_threads;
  int which_dif_param;
  bool bipartite;
  double lambda;
  bool rmse_fit;
  int which_target_metric;
  bool itm_trsp;
  bool save_vals;
  bool set_trend;
  int best_step;
} ctrl_t;

// struct forcommand line arguments
typedef struct {
  input_t input;
  output_t output;
  ctrl_t ctrl;
} cmd_args_t;

// Csr graph struct
typedef struct {
  long double *csr_value;
  sz_long_t *csr_column;
  sz_long_t *csr_row_pointer;
  sz_long_t num_nodes;
  sz_long_t nnz;
  sz_long_t *degrees;
  char name[30];
} csr_graph_t;

// Double matrix
typedef struct {
  double **val;
  sz_long_t num_rows;
  sz_long_t num_cols;
} d_mat_t;

// Integer matrix
typedef struct {
  sz_long_t **val;
  sz_long_t num_rows;
  sz_long_t num_cols;
} i_mat_t;

// Structure that holds data
typedef struct {
  sz_long_t num_users;
  int num_models;
  user_t *users;
  user_t *users_full;
  csr_graph_t *item_models;
  i_mat_t cv_neg;
  sz_long_t *cv_pos;
  i_mat_t test_neg;
  sz_long_t *test_pos;
  bool no_models;
} data_t;

// Rating quality metrics
typedef struct {
  double HR;
  double ARHR;
  double NDCG;
} metric_t;

// Output struct that contaions thetas and predictions
typedef struct {
  int num_models;
  sz_long_t num_users;
  d_mat_t *thetas;
  d_mat_t *mus;
  metric_t **test_result;
  d_mat_t *test_pred;
  d_mat_t *val_pred;
  metric_t **general_trends;
  metric_t ***local_trend;
} out_t;

// Report parameters for threads
typedef struct {
  bool yes;
  int every;
  int shift;
} report_t;

// type to pass to threads (either for training or prediction)
struct pass2thread_t {
  void (*func)(struct pass2thread_t, int);
  int m_id;
  int u_id;
  int model_start;
  int model_win;
  sz_long_t usr_start;
  sz_long_t usr_win;
  data_t data;
  csr_graph_t **graphs;
  out_t out;
  ctrl_t ctrl;
  report_t report;
};

#endif
