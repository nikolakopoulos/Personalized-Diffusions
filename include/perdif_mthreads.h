#ifndef PERDIF_MTHREADS_H_
#define PERDIF_MTHREADS_H_

#include "rec_defs.h"

void distribute_to_threads(csr_graph_t *, data_t , out_t, ctrl_t, void (*func)( struct pass2thread_t, int ) );

#endif
