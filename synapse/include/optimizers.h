#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

typedef struct {
    float *w_momentum;
    float *b_momentum;
    float *w_squared_grads;
    float *b_squared_grads;
    int w_start;
    int b_start;
} OptimizerCache;

void set_beta1(float new_beta1);
void set_beta2(float new_beta2);

OptimizerCache* init_optimizer_cache(int (*optimizer)(float *restrict, const float *restrict, int, float, OptimizerCache *, int),
    int num_weights,
    int num_biases
);

void free_optimizer_cache(OptimizerCache **cache);

int sgd(float *restrict weights,
    const float *restrict weight_grads,
    int size,
    float learning_rate,
    OptimizerCache *cache,
    int flag
);

int momentum(float *restrict weights,
    const float *restrict weight_grads,
    int size,
    float learning_rate,
    OptimizerCache *cache,
    int flag
);

int adagrad(float *restrict weights,
    const float *restrict weight_grads,
    int size,
    float learning_rate,
    OptimizerCache *cache,
    int flag
);

int rmsprop(float *restrict weights,
    const float *restrict weight_grads,
    int size,
    float learning_rate,
    OptimizerCache *cache,
    int flag
);

int adam(float *restrict weights,
    const float *restrict weight_grads,
    int size,
    float learning_rate,
    OptimizerCache *cache,
    int flag
);

#endif
