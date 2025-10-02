#ifndef ACTIV_FUNCS_H
#define ACTIV_FUNCS_H

int linear(const float *restrict x, float *restrict out, int size);
int relu(const float *restrict x, float *restrict out, int size);
int sigmoid(const float *restrict x, float *restrict out, int size);
int softmax(const float *restrict x, float *restrict out, int size);

float grad_activ_func(int (*activ_func)(const float *restrict, float *restrict, int), float x);

#endif
