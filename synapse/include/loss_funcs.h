#ifndef LOSS_FUNCS_H
#define LOSS_FUNCS_H

float mean_squared_error(const float *restrict y_true, const float *restrict y_pred, int size);
float binary_cross_entropy(const float *restrict y_true, const float *restrict y_pred, int size);
float categorical_cross_entropy(const float *restrict y_true, const float *restrict y_pred, int size);

#endif
