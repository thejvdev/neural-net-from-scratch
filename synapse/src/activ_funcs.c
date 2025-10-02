#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "activ_funcs.h"


#define CHECK_ACTIV_ARGS(x, out, size) \
    if (!(x) || !(out) || (size) <= 0) { \
        fprintf(stderr, "Error in %s(): Invalid input parameters.\n", __func__); \
        return 1; \
    }


int linear(const float *restrict x, float *restrict out, int size) {
    CHECK_ACTIV_ARGS(x, out, size);
    memcpy(out, x, size * sizeof(float));
    return 0;
}


int relu(const float *restrict x, float *restrict out, int size) {
    CHECK_ACTIV_ARGS(x, out, size);
    
    for (int i = 0; i < size; ++i) {
        out[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }

    return 0;
}


int sigmoid(const float *restrict x, float *restrict out, int size) {
    CHECK_ACTIV_ARGS(x, out, size);

    for (int i = 0; i < size; ++i) {
        out[i] = 1.0f / (1.0f + expf(-x[i]));
    }

    return 0;
}


int softmax(const float *restrict x, float *restrict out, int size) {
    CHECK_ACTIV_ARGS(x, out, size);

    float max = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        out[i] = expf(x[i] - max);
        sum += out[i];
    }
    
    for (int i = 0; i < size; ++i) {
        out[i] /= sum;
    }

    return 0;
}


float grad_activ_func(int (*activ_func)(const float *restrict, float *restrict, int), float x) {
    if (!activ_func) {
        fprintf(stderr, "Error in %s(): Invalid input parameters.\n", __func__);
        return NAN;
    }
    
    float res = 0.0f;

    if (activ_func == linear) {
        res = 1;
    } else if (activ_func == relu) {
        res = (x > 0.0f) ? 1.0f : 0.0f;
    } else if (activ_func == sigmoid) {
        float s = 1.0f / (1.0f + expf(-x));
        res = s * (1.0f - s);
    }

    return res;
}
