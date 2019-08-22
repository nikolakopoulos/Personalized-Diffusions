#ifndef PERDIF_PREDICT_H_
#define PERDIF_PREDICT_H_

#include "rec_defs.h"

void perdif_predict(data_t, out_t, ctrl_t, output_t);

void perdif_predict_smodel(struct pass2thread_t, int);

#endif
