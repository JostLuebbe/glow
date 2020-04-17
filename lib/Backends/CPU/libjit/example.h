#ifndef GLOW_EXAMPLE_H
#define GLOW_EXAMPLE_H

#include "libjit_dim_t.h"

void glow_conv(int8_t *result, const int8_t *inW, const int8_t *filterW,
               const int32_t *biasW, const dim_t *outWdims,
               const dim_t *inWdims, const dim_t *filterWdims,
               const dim_t *biasWdims, int32_t outOffset, int32_t inOffset,
               int32_t filterOffset, int32_t biasOffset, int32_t biasPre,
               int32_t biasPost, int32_t biasScale, int32_t outPre,
               int32_t outPost, int32_t outScale);

#endif //GLOW_EXAMPLE_H
