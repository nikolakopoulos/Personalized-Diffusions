#ifndef REC_IO_H_
#define REC_IO_H_

#include "rec_defs.h"

void parse_commandline_args(int, char **, cmd_args_t *);

user_t *read_rating_matrix_csr(sz_long_t *, char *);

csr_graph_t read_item_graph_csr(char *, char *);

csr_graph_t *read_item_models(char *, int *);

void write_d_mat(d_mat_t, char *);

data_t load_data(input_t, ctrl_t *);

data_t load_CV_data(input_t, ctrl_t *);

void store_general_trends(metric_t **, char *, int, int, bool, const char **);

void store_vals(char *, d_mat_t *, int, bool);

void store_parameters(d_mat_t *, d_mat_t *, char *, int, const char **);

void store_results(metric_t **, char *, int, sz_long_t, const char **);

void compute_mus(d_mat_t *, d_mat_t *, int);

void green();

void red();

void yellow();

void magenta();

void blue();

void cyan();

void colorreset();

#endif
