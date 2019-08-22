#ifndef CSR_HANDLING_H_
#define CSR_HANDLING_H_

#include "rec_defs.h"

void make_BCSR_col_stoch(csr_graph_t*);

csr_graph_t csr_create( const sz_long_t** , sz_long_t);

csr_graph_t csr_deep_copy_and_scale(csr_graph_t, double );

csr_graph_t* csr_mult_deep_copy( csr_graph_t, sz_short_t );

csr_graph_t csr_deep_copy(csr_graph_t);

void my_CSR_matmat( double* Y ,double* X  , csr_graph_t , sz_med_t , sz_med_t , sz_med_t);

void my_CSR_matvec( double* ,double* ,csr_graph_t);

csr_graph_t csr_transpose( csr_graph_t );

sz_long_t** csr_to_edgelist(csr_graph_t , long double**);

csr_graph_t edgelist_to_csr(sz_long_t**, long double*, sz_long_t, sz_long_t);

void csr_rec_normalize(csr_graph_t* );

void csr_add_diagonal(csr_graph_t*, double* );

#endif
