#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "loss_funcs.h"
#include "utils.h"


#define CHECK_LOSS_ARGS(y_true, y_pred, size) \
    if (!(y_true) || !(y_pred) || (size) <= 0) { \
        fprintf(stderr, "Error in %s(): Invalid input parameters.\n", __func__); \
        return NAN; \
    }


float mean_squared_error(const float *restrict y_true, const float *restrict y_pred, int size) {
    CHECK_LOSS_ARGS(y_true, y_pred, size);
    
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        float error = y_true[i] - y_pred[i];
        sum += error * error;
    }

    return 0.5f * sum / size;
}


float binary_cross_entropy(const float *restrict y_true, const float *restrict y_pred, int size) {
    CHECK_LOSS_ARGS(y_true, y_pred, size);

    float sum = 0.0f;

    for (int i = 0; i < size; ++i) {
        float clipped_pred = fmaxf(EPSILON, fminf(1.0f - EPSILON, y_pred[i]));
        sum += y_true[i] * logf(clipped_pred) + (1.0f - y_true[i]) * logf(1.0f - clipped_pred);
    }

    return -sum / size;
}


float categorical_cross_entropy(const float *restrict y_true, const float *restrict y_pred, int size) {
    CHECK_LOSS_ARGS(y_true, y_pred, size);

    int class_index = -1;
    for (int i = 0; i < size; ++i) {
        if (y_true[i] == 1.0f) {
            class_index = i;
            break;
        }
    }

    if (class_index == -1) {
        fprintf(stderr, "Error in %s: Invalid class index.\n", __func__);
        return NAN;
    }
    
    return -logf(fmaxf(y_pred[class_index], EPSILON));
}
