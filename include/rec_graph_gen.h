#ifndef REC_GRAPH_GEN_H_
#define REC_GRAPH_GEN_H_

#include "rec_defs.h"

csr_graph_t generate_rec_graph(user_t *users, csr_graph_t item_model,
                               sz_long_t num_users, bool bipartite);

#endif
