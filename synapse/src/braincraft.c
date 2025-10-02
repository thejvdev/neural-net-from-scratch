#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "braincraft.h"
#include "utils.h"


typedef struct {
    int input_size;
    int output_size;
    float *weights;
    float *weight_grads;
    float *biases;
    float *bias_grads;
    float *deltas;
    float *sums;
    float *activs;
    int (*activ_func)(const float *restrict, float *restrict, int);
} Layer;


static Layer *_nn = NULL;
static int _num_layers = 0;
static int _lidx = 0;
static int _num_weights = 0;
static int _num_biases = 0;

static float (*_loss_func)(const float *restrict, const float *restrict, int) = NULL;
static int (*_optimizer)(float *restrict, const float *restrict, int, float, OptimizerCache *, int) = NULL;
static OptimizerCache *_cache = NULL;
static float _learning_rate = 0.0f;


int create_neural_network(int num_layers) {
    if (_nn) {
        fprintf(stderr, "Error: Neural network already created.\n");
        return 1;
    }

    if (num_layers <= 0) {
        fprintf(stderr, "Error: Invalid number of layers specified for the neural network.\n");
        return 1;
    }

    _nn = (Layer *)malloc((_num_layers = num_layers) * sizeof(Layer));
    if (!_nn) {
        fprintf(stderr, "Error: Memory allocation failed for neural network layers.\n");
        return 1;
    }

    return 0;
}


void delete_neural_network(void) {
    if (!_nn) return;

    free_optimizer_cache(&_cache);

    for (int i = 0; i < _num_layers; ++i) {
        Layer *layer = &_nn[i];

        if (layer->weights) {
            free(layer->weights);
            layer->weights = NULL;
        }

        if (layer->weight_grads) {
            free(layer->weight_grads);
            layer->weight_grads = NULL;
        }

        if (layer->biases) {
            free(layer->biases);
            layer->biases = NULL;
        }

        if (layer->bias_grads) {
            free(layer->bias_grads);
            layer->bias_grads = NULL;
        }

        if (layer->deltas) {
            free(layer->deltas);
            layer->deltas = NULL;
        }

        if (layer->sums) {
            free(layer->sums);
            layer->biases = NULL;
        }
        
        if (layer->activs) {
            free(layer->activs);
            layer->biases = NULL;
        }
    }

    free(_nn);
    _nn = NULL;
}


const char* get_activ_func_name(int (*activ_func)(const float *restrict, float *restrict, int)) {
    if (activ_func == linear) return "Linear";
    if (activ_func == relu) return "ReLU";
    if (activ_func == sigmoid) return "Sigmoid";
    if (activ_func == softmax) return "Softmax";
    return "Unknown";
}


int (*get_activ_func_by_name(const char *name))(const float *restrict, float *restrict, int) {
    if (strcmp(name, "Linear") == 0) return linear;
    if (strcmp(name, "ReLU") == 0) return relu;
    if (strcmp(name, "Sigmoid") == 0) return sigmoid;
    if (strcmp(name, "Softmax") == 0) return softmax;
    return NULL;
}


void info_neural_network(void) {
    if (!_nn || _num_layers != _lidx) return;

    for (int l = 0; l < ((_num_layers <= 10) ? _num_layers : 10); ++l) {
        Layer *layer = &_nn[l];
        const char *activ_func_name = get_activ_func_name(layer->activ_func);

        printf("Layer: %d Input size: %d Output size: %d\n\n", l + 1, layer->input_size, layer->output_size);

        for (int i = 0; i < ((layer->output_size <= 10) ? layer->output_size : 10); ++i) {
            printf("  Neuron %d:\n", i + 1);
            
            int limit = (layer->input_size < 10) ? layer->input_size : 10;

            printf("             Weights:  [");
            for (int j = 0; j < limit; ++j) {
                printf(" %f", layer->weights[i * layer->input_size + j]);
            }
            if (layer->input_size > 10) printf(" ...");
            printf(" ]\n");

            printf("                Bias:  %f\n", layer->biases[i]);

            printf("    Weight gradients:  [");            
            for (int j = 0; j < limit; ++j) {
                printf(" %f", layer->weight_grads[i * layer->input_size + j]);
            }
            if (layer->input_size > 10) printf(" ...");
            printf(" ]\n");

            printf("       Bias gradient:  %f\n", layer->bias_grads[i]);
            printf("                 Sum:  %f\n", layer->sums[i]);
            printf("          Activation:  %f\n\n", layer->activs[i]);
        }

        if (layer->output_size > 10) printf("  ...\n\n");

        printf("  Activation function: %s\n", activ_func_name);
        printf("  Number of weights:   %d\n", layer->input_size * layer->output_size);
        printf("  Number of biases:    %d\n\n", layer->output_size);
    }
}


int save_neural_network(const char *filename) {
    if (!_nn) {
        fprintf(stderr, "Error: Neural network not created.\n");
        return 1;
    }

    if (_num_layers != _lidx) {
        fprintf(stderr, "Error: Neural network not properly initialized.\n");
        return 1;
    }

    if (!filename) {
        fprintf(stderr, "Error: Invalid file name provided.\n");
        return 1;
    }

    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file '%s'.\n", filename);
        return 1;
    }

    fwrite(&_num_layers, sizeof(int), 1, file);

    for (int l = 0; l < _num_layers; ++l) {
        Layer *layer = &_nn[l];

        fwrite(&layer->input_size, sizeof(int), 1, file);
        fwrite(&layer->output_size, sizeof(int), 1, file);

        int num_weights = layer->input_size * layer->output_size;
        fwrite(layer->weights, sizeof(float), num_weights, file);
        fwrite(layer->biases, sizeof(float), layer->output_size, file);

        const char *activ_func_name = get_activ_func_name(layer->activ_func);
        int name_len = (int)strlen(activ_func_name) + 1;
        fwrite(&name_len, sizeof(int), 1, file);
        fwrite(activ_func_name, sizeof(char), name_len, file);
    }

    fclose(file); 
    return 0;
}


int load_neural_network(const char *filename) {
    if (!filename) {
        fprintf(stderr, "Error: Invalid file name provided.\n");
        return 1;
    }

    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file '%s'.\n", filename);
        return 1;
    }
    
    fread(&_num_layers, sizeof(int), 1, file);
    _lidx = _num_layers;

    create_neural_network(_num_layers);

    for (int l = 0; l < _num_layers; ++l) {
        Layer *layer = &_nn[l];

        fread(&layer->input_size, sizeof(int), 1, file);
        fread(&layer->output_size, sizeof(int), 1, file);

        int num_weights = layer->input_size * layer->output_size;
        layer->weights = (float *)malloc(sizeof(float) * num_weights);
        fread(layer->weights, sizeof(float), num_weights, file);

        layer->biases = (float *)malloc(sizeof(float) * layer->output_size);
        fread(layer->biases, sizeof(float), layer->output_size, file);
        
        int name_len;
        fread(&name_len, sizeof(int), 1, file);
        char *activ_func_name = (char *)malloc(name_len);
        fread(activ_func_name, sizeof(char), name_len, file);

        layer->activ_func = get_activ_func_by_name(activ_func_name);
        free(activ_func_name);
        
        int output_size = layer->output_size;
        layer->weight_grads = (float *)calloc(num_weights, sizeof(float));
        layer->bias_grads = (float *)calloc(output_size, sizeof(float));
        layer->deltas = (float *)malloc(output_size * sizeof(float));
        layer->sums = (float *)calloc(output_size, sizeof(float));
        layer->activs = (float *)calloc(output_size, sizeof(float));

        _num_weights += num_weights;
        _num_biases += output_size;

        if (!layer->weights || !layer->weight_grads || !layer->biases || !layer->bias_grads ||
            !layer->deltas || !layer->sums || !layer->activs
        ) {
            fprintf(stderr, "Error: Memory allocation failed.\n");
            fclose(file);
            return 1;
        }
    }

    fclose(file);
    return 0;
}


float rand_normal(float mean, float stddev) {
    static int have_spare = 0;
    static float spare;
    if (have_spare) {
        have_spare = 0;
        return mean + stddev * spare;
    }

    have_spare = 1;
    float u, v, s;
    do {
        u = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        v = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);

    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    return mean + stddev * (u * s);
}


void init_weights(float *weights, int input_size, int output_size,
    int (*activ_func)(const float *restrict, float *restrict, int)
) {
    float gain = 1.0f;

    if (activ_func == relu) {
        gain = sqrtf(2.0f);
    } else if (activ_func == linear || activ_func == sigmoid || activ_func == softmax) {
        gain = 1.0f;
    } 
    
    float stddev = gain / sqrtf(input_size);

    int size = input_size * output_size;
    for (int i = 0; i < size; ++i) {
        weights[i] = rand_normal(0.0f, stddev);
    }
}


int init_layer(int input_size, int output_size, int (*activ_func)(const float *restrict, float *restrict, int)) {
    if (!_nn) {
        fprintf(stderr, "Error: Neural network not created.\n");
        return 1;
    }

    if (_lidx >= _num_layers) {
        fprintf(stderr, "Error: Neural network already initialized.\n");
        return 1;
    }

    if (input_size <= 0 || output_size <= 0 || !activ_func) {
        fprintf(stderr, "Error: Invalid input parameters for layer %d.\n", _lidx + 1);
        return 1;
    }

    Layer *layer = &_nn[_lidx++];

    layer->weights = (float *)malloc(input_size * output_size * sizeof(float));
    layer->weight_grads = (float *)calloc(input_size * output_size, sizeof(float));
    layer->biases = (float *)calloc(output_size, sizeof(float));
    layer->bias_grads = (float *)calloc(output_size, sizeof(float));
    layer->deltas = (float *)malloc(output_size * sizeof(float));
    layer->sums = (float *)calloc(output_size, sizeof(float));
    layer->activs = (float *)calloc(output_size, sizeof(float));

    if (!layer->weights || !layer->weight_grads || !layer->biases || !layer->bias_grads ||
        !layer->deltas || !layer->sums || !layer->activs
    ) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return 1;
    }

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activ_func = activ_func;

    init_weights(layer->weights, input_size, output_size, activ_func);
    
    _num_weights += input_size * output_size;
    _num_biases += output_size;

    return 0;
}


int setup_loss_function(float (*loss_func)(const float *restrict, const float *restrict, int)) {
    if (!loss_func) {
        fprintf(stderr, "Error: Invalid input parameters for setting up the loss function.\n");
        return 1;
    }

    _loss_func = loss_func;

    return 0;
}


int setup_optimizer(int (*optimizer)(float *restrict, const float *restrict, int, float, OptimizerCache *, int),
    float learning_rate
) {
    if (!_nn) {
        fprintf(stderr, "Error: Neural network not created.\n");
        return 1;
    }

    if (_num_layers != _lidx) {
        fprintf(stderr, "Error: Neural network not properly initialized.\n");
        return 1;
    }

    if (!optimizer || learning_rate <= 0.0f || learning_rate > 0.1f) {
        fprintf(stderr, "Error: Invalid input parameters for setting up the optimizer.\n");
        return 1;
    }

    _optimizer = optimizer;
    _learning_rate = learning_rate;

    _cache = init_optimizer_cache(optimizer, _num_weights, _num_biases);

    return 0;
}


float* forward(const float *inputs) {
    if (!_nn) {
        fprintf(stderr, "Error: Neural network not created.\n");
        return NULL;
    }

    if (_num_layers != _lidx) {
        fprintf(stderr, "Error: Neural network not properly initialized.\n");
        return NULL;
    }

    if (!inputs) {
        fprintf(stderr, "Error: Invalid input parameters for the forward pass.\n");
        return NULL;
    }

    for (int l = 0; l < _num_layers; ++l) {
        Layer *prev_layer = (l >= 1) ? &_nn[l - 1] : NULL;
        Layer *layer = &_nn[l];
        
        int input_size = layer->input_size;
        int output_size = layer->output_size;
        
        float *sums = layer->sums;
        memset(sums, 0, output_size * sizeof(float));
        
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                float x = (l >= 1) ? prev_layer->activs[j] : inputs[j];
                sums[i] += x * layer->weights[i * input_size + j];
            }
            sums[i] += layer->biases[i];
        }
        
        layer->activ_func(sums, layer->activs, output_size);
    }

    return _nn[_num_layers - 1].activs;
}


float compute_loss(const float *y_true) {
    if (!_nn) {
        fprintf(stderr, "Error: Neural network not created.\n");
        return NAN;
    }

    if (_num_layers != _lidx) {
        fprintf(stderr, "Error: Neural network not properly initialized.\n");
        return NAN;
    }
    
    if (!_loss_func) {
        fprintf(stderr, "Error: Loss function not initialized.\n");
        return NAN;
    }

    if (!y_true) {
        fprintf(stderr, "Error: Invalid input parameters for computing the loss function.\n");
        return NAN;
    }

    return _loss_func(y_true, _nn[_num_layers - 1].activs, _nn[_num_layers - 1].output_size);
}


int compute_output_grads(Layer *layer, const float *restrict prev_activs, const float *restrict y_true) {
    const int input_size = layer->input_size;
    const int output_size = layer->output_size;

    float *deltas = layer->deltas;
    float *weight_grads = layer->weight_grads;
    float *bias_grads = layer->bias_grads;

    const float *sums = layer->sums;
    const float *activs = layer->activs;

    if ((_loss_func == categorical_cross_entropy && layer->activ_func == softmax) || 
        (_loss_func == binary_cross_entropy && layer->activ_func == sigmoid)
    ) {
        for (int i = 0; i < output_size; ++i) {
            deltas[i] = activs[i] - y_true[i];
        }
    } else if (_loss_func == mean_squared_error && layer->activ_func != softmax) {
        for (int i = 0; i < output_size; ++i) {
            deltas[i] = (activs[i] - y_true[i]) * grad_activ_func(layer->activ_func, sums[i]);
        }
    } else {
        fprintf(stderr, "Error: Failed to compute deltas in the output layer.\n");
        return 1;
    }
    
    for (int i = 0; i < output_size; ++i) {
        float delta = deltas[i];
        for (int j = 0; j < input_size; ++j) {
            weight_grads[i * input_size + j] += delta * prev_activs[j];
        }
        bias_grads[i] += delta;
    }

    return 0;
}


int compute_inner_grads(Layer *restrict layer, Layer *restrict next_layer, const float *prev_activs) {
    if (layer->activ_func == softmax) {
        fprintf(stderr, "Error: Failed to compute gradients in the hidden layers.\n");
        return 1;
    }

    const int input_size = layer->input_size;
    const int output_size = layer->output_size;
    
    float *deltas = layer->deltas;
    float *weight_grads = layer->weight_grads;
    float *bias_grads = layer->bias_grads;

    const float *sums = layer->sums;
    const float *next_deltas = next_layer->deltas;
    const float *next_weights = next_layer->weights;
    
    for (int i = 0; i < output_size; ++i) {
        float weighted_sum = 0.0f;

        for (int j = 0; j < next_layer->output_size; ++j) {
            weighted_sum += next_deltas[j] * next_weights[j * next_layer->input_size + i];
        }

        deltas[i] = weighted_sum * grad_activ_func(layer->activ_func, sums[i]);
    }
    
    for (int i = 0; i < output_size; ++i) {
        float delta = deltas[i];
        for (int j = 0; j < input_size; ++j) {
            weight_grads[i * input_size + j] += delta * prev_activs[j];
        }
        bias_grads[i] += delta;
    }

    return 0;
}


int backward(const float *restrict inputs, const float *restrict y_true) {
    if (!_nn) {
        fprintf(stderr, "Error: Neural network not created.\n");
        return 1;
    }

    if (_num_layers != _lidx) {
        fprintf(stderr, "Error: Neural network not properly initialized.\n");
        return 1;
    }

    if (!_loss_func) {
        fprintf(stderr, "Error: Loss function not initialized.\n");
        return 1;
    }

    if (!inputs || !y_true) {
        fprintf(stderr, "Error: Invalid input parameters for the backward pass.\n");
        return 1;
    }
    
    if (compute_output_grads(&_nn[_num_layers - 1], ((_num_layers > 1) ? _nn[_num_layers - 2].activs : inputs), y_true)) {
        return 1;
    }

    for (int l = _num_layers - 2; l >= 0; --l) {
        if (compute_inner_grads(&_nn[l], &_nn[l + 1], ((l > 0) ? _nn[l - 1].activs : inputs))) {
            return 1;
        }
    }

    return 0;
}


int update_weights(void) {
    if (!_nn) {
        fprintf(stderr, "Error: Neural network not created.\n");
        return 1;
    }

    if (_num_layers != _lidx) {
        fprintf(stderr, "Error: Neural network not properly initialized.\n");
        return 1;
    }

    if (!_optimizer) {
        fprintf(stderr, "Error: Optimizer not initialized.\n");
        return 1;
    }
    
    if (_cache) {
        _cache->w_start = 0;
        _cache->b_start = 0;
    }
    
    for (int l = 0; l < _num_layers; ++l) {
        Layer *layer = &_nn[l];
        int num_weights = layer->input_size * layer->output_size;
        int num_biases = layer->output_size;

        _optimizer(layer->weights, layer->weight_grads, num_weights, _learning_rate, _cache, 1);
        _optimizer(layer->biases, layer->bias_grads, num_biases, _learning_rate, _cache, 0);

        if (_cache) {
            _cache->w_start += num_weights;
            _cache->b_start += num_biases;
        }
    }

    return 0;
}


int zero_grads(void) {
    if (!_nn) {
        fprintf(stderr, "Error: Neural network not created.\n");
        return 1;
    }

    if (_num_layers != _lidx) {
        fprintf(stderr, "Error: Neural network not properly initialized.\n");
        return 1;
    }
    
    for (int l = 0; l < _num_layers; ++l) {
        Layer *layer = &_nn[l];
        int num_weights = layer->input_size * layer->output_size;
        memset(layer->weight_grads, 0, num_weights * sizeof(float));
        memset(layer->bias_grads, 0, layer->output_size * sizeof(float));
    }
    
    return 0;
}
