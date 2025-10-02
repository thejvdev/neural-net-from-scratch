#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "optimizers.h"
#include "utils.h"


#define CHECK_OPTIM_ARGS(weights, weight_grads, size, learning_rate) \
    if (!(weights) || !(weight_grads) || (size) <= 0 || (learning_rate) <= 0.0f || (learning_rate) > 0.1f) { \
        fprintf(stderr, "Error in %s(): Invalid input parameters.\n", __func__); \
        return 1; \
    }

static float _beta1 = 0.9f;
static float _beta2 = 0.999f;


void set_beta1(float new_beta1) {
    if (new_beta1 < 0.0f || new_beta1 >= 1.0f) {
        fprintf(stderr, "Error: beta1 should be in the range [0, 1).\n");
        return;
    }
    _beta1 = new_beta1;
}


void set_beta2(float new_beta2) {
    if (new_beta2 < 0.0f || new_beta2 >= 1.0f) {
        fprintf(stderr, "Error: beta2 should be in the range [0, 1).\n");
        return;
    }
    _beta2 = new_beta2;
}


void free_optimizer_cache(OptimizerCache **cache) {
    if (!cache || !*cache) return;

    if ((*cache)->w_momentum) {
        free((*cache)->w_momentum);
        (*cache)->w_momentum = NULL;
    }

    if ((*cache)->b_momentum) {
        free((*cache)->b_momentum);
        (*cache)->b_momentum = NULL;
    }

    if ((*cache)->w_squared_grads) {
        free((*cache)->w_squared_grads);
        (*cache)->w_squared_grads = NULL;
    }

    if ((*cache)->b_squared_grads) {
        free((*cache)->b_squared_grads);
        (*cache)->b_squared_grads = NULL;
    }

    free(*cache);
    *cache = NULL;
}


OptimizerCache* init_optimizer_cache(int (*optimizer)(float *restrict, const float *restrict, int, float, OptimizerCache *, int),
    int num_weights,
    int num_biases
) {
    if (!optimizer || num_weights <= 0 || num_biases <= 0) {
        fprintf(stderr, "Error: Invalid input parameters for optimizer cache.\n");
        return NULL;
    }

    if (optimizer == sgd) return NULL;

    OptimizerCache *cache = (OptimizerCache *)malloc(sizeof(OptimizerCache));
    if (!cache) {
        fprintf(stderr, "Error: Memory allocation failed for the optimizer cache.\n");
        return NULL;
    }

    cache->w_momentum = NULL;
    cache->b_momentum = NULL;

    cache->w_squared_grads = NULL;
    cache->b_squared_grads = NULL;

    cache->w_start = 0;
    cache->b_start = 0;

    if (optimizer == momentum) {
        cache->w_momentum = (float *)calloc(num_weights, sizeof(float));
        cache->b_momentum = (float *)calloc(num_biases, sizeof(float));

        if (!cache->w_momentum || !cache->b_momentum) goto cleanup;

    } else if (optimizer == adagrad || optimizer == rmsprop) {
        cache->w_squared_grads = (float *)calloc(num_weights, sizeof(float));
        cache->b_squared_grads = (float *)calloc(num_biases, sizeof(float));

        if (!cache->w_squared_grads || !cache->b_squared_grads) goto cleanup;

    } else if (optimizer == adam) {
        cache->w_momentum = (float *)calloc(num_weights, sizeof(float));
        cache->b_momentum = (float *)calloc(num_biases, sizeof(float));

        cache->w_squared_grads = (float *)calloc(num_weights, sizeof(float));
        cache->b_squared_grads = (float *)calloc(num_biases, sizeof(float));

        if (!cache->w_momentum || !cache->b_momentum || !cache->w_squared_grads || !cache->b_squared_grads) {
            goto cleanup;
        }

    } else {
        free(cache);
        cache = NULL;
    }

    return cache;

cleanup:
    free_optimizer_cache(&cache);
    return NULL;
}


int sgd(float *restrict weights,
    const float *restrict weight_grads,
    int size,
    float learning_rate,
    OptimizerCache *cache,
    int flag
) {
    CHECK_OPTIM_ARGS(weights, weight_grads, size, learning_rate);

    (void)cache;
    (void)flag;
    
    for (int i = 0; i < size; ++i) {
        weights[i] -= learning_rate * weight_grads[i];
    }

    return 0;
}


int momentum(float *restrict weights,
    const float *restrict weight_grads,
    int size,
    float learning_rate,
    OptimizerCache *cache,
    int flag
) {
    CHECK_OPTIM_ARGS(weights, weight_grads, size, learning_rate);

    int start = 0;
    float *momentum = NULL;

    if (flag) {
        start = cache->w_start;
        momentum = cache->w_momentum;
    } else {
        start = cache->b_start;
        momentum = cache->b_momentum;
    }
    
    for (int i = 0; i < size; ++i) {
        momentum[start + i] = momentum[start + i] * _beta1 + (1 - _beta1) * weight_grads[i];
        weights[i] -= learning_rate * momentum[start + i];
    }

    return 0;
}


int adagrad(float *restrict weights,
    const float *restrict weight_grads,
    int size,
    float learning_rate,
    OptimizerCache *cache,
    int flag
) {
    CHECK_OPTIM_ARGS(weights, weight_grads, size, learning_rate);

    int start = 0;
    float *squared_grads = NULL;

    if (flag) {
        start = cache->w_start;
        squared_grads = cache->w_squared_grads;
    } else {
        start = cache->b_start;
        squared_grads = cache->b_squared_grads;
    }
    
    for (int i = 0; i < size; ++i) {
        squared_grads[start + i] += weight_grads[i] * weight_grads[i];
        weights[i] -= learning_rate * weight_grads[i] / (sqrtf(squared_grads[start + i]) + EPSILON);
    }

    return 0;
}


int rmsprop(float *restrict weights,
    const float *restrict weight_grads,
    int size,
    float learning_rate,
    OptimizerCache *cache,
    int flag
) {
    CHECK_OPTIM_ARGS(weights, weight_grads, size, learning_rate);

    int start = 0;
    float *squared_grads = NULL;

    if (flag) {
        start = cache->w_start;
        squared_grads = cache->w_squared_grads;
    } else {
        start = cache->b_start;
        squared_grads = cache->b_squared_grads;
    }
    
    for (int i = 0; i < size; ++i) {
        squared_grads[start + i] = squared_grads[start + i] * _beta2 + (1 - _beta2) * weight_grads[i] * weight_grads[i];
        weights[i] -= learning_rate * weight_grads[i] / (sqrtf(squared_grads[start + i]) + EPSILON);
    }

    return 0;
}


int adam(float *restrict weights,
    const float *restrict weight_grads,
    int size,
    float learning_rate,
    OptimizerCache *cache,
    int flag
) {
    CHECK_OPTIM_ARGS(weights, weight_grads, size, learning_rate);

    static int t = 0;
    t += 1;

    int start = 0;
    float *momentum = NULL;
    float *squared_grads = NULL;

    if (flag) {
        start = cache->w_start;
        momentum = cache->w_momentum;
        squared_grads = cache->w_squared_grads;
    } else {
        start = cache->b_start;
        momentum = cache->b_momentum;
        squared_grads = cache->b_squared_grads;
    }

    for (int i = 0; i < size; ++i) {
        momentum[start + i] = momentum[start + i] * _beta1 + (1 - _beta1) * weight_grads[i];
        squared_grads[start + i] = squared_grads[start + i] * _beta2 + (1 - _beta2) * weight_grads[i] * weight_grads[i];

        float m_hat = momentum[start + i] / (1 - powf(_beta1, t));
        float v_hat = squared_grads[start + i] / (1 - powf(_beta2, t));

        weights[i] -= learning_rate * m_hat / (sqrtf(v_hat) + EPSILON);
    }

    return 0;
}
