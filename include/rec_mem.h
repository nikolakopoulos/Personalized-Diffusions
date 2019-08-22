#ifndef REC_MEM_H_
#define REC_MEM_H_

#include "rec_defs.h"

void csr_alloc(csr_graph_t* );

void csr_realloc(csr_graph_t* , sz_long_t , sz_long_t );

void csr_free( csr_graph_t );

void csr_array_free( csr_graph_t* , sz_short_t);

d_mat_t d_mat_init( sz_long_t , sz_long_t );

void d_mat_free(d_mat_t );

i_mat_t i_mat_init( sz_long_t , sz_long_t );

void i_mat_free(i_mat_t );

void usr_free(user_t* , sz_long_t );

void data_free(data_t);

void d_mat_resize( d_mat_t* , sz_long_t , sz_long_t );

void i_mat_resize( i_mat_t* , sz_long_t , sz_long_t );

out_t output_init(data_t , ctrl_t);

void output_free(out_t, ctrl_t);

metric_t ***init_local_trend(int , int , int);

void free_local_trend( metric_t***, int , int );

data_t data_init(void);

#endif
