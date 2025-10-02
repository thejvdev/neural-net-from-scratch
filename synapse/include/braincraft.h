#ifndef BRAINCRAFT_H
#define BRAINCRAFT_H

#include "activ_funcs.h"
#include "loss_funcs.h"
#include "optimizers.h"

int create_neural_network(int num_layers);
void delete_neural_network(void);
void info_neural_network(void);
int save_neural_network(const char *filename);
int load_neural_network(const char *filename);

int init_layer(int input_size, int output_size, int (*activ_func)(const float *restrict, float *restrict, int));

int setup_loss_function(float (*loss_func)(const float *restrict, const float *restrict, int));
int setup_optimizer(int (*optimizer)(float *restrict, const float *restrict, int, float, OptimizerCache *, int), 
    float learning_rate
);

float* forward(const float *inputs);
float compute_loss(const float *y_true);
int backward(const float *restrict inputs, const float *restrict y_true);
int update_weights(void);
int zero_grads(void);

#endif
