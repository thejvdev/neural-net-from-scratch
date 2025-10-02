#include <stdio.h>
#include <stdlib.h>

#include "loader.h"


#define CHECK_LOAD_ARGS(filename, num_samples, input_size) \
    if (!(filename) || (num_samples) <= 0 || (input_size) <= 0) { \
        fprintf(stderr, "Error in %s(): Invalid input parameters.\n", __func__); \
        return NULL; \
    }

#define DELETE_DATA_LABELS(data, num_samples) \
    do { \
        if ((data)) { \
            for (int i = 0; i < (num_samples); ++i) { \
                if ((data)[i]) free((data)[i]); \
            } \
            free((data)); \
        } \
    } while (0);


float** read_csv_data(const char *filename, int num_samples, int input_size) {
    CHECK_LOAD_ARGS(filename, num_samples, input_size);

    float **data = (float **)malloc(num_samples * sizeof(float *));
    for (int i = 0; i < num_samples; ++i) {
        data[i] = (float *)malloc(input_size * sizeof(float));
    }

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file '%s'.\n", filename);
        DELETE_DATA_LABELS(data, num_samples);
        return NULL;
    }

    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < input_size; ++j) {
            if (fscanf(file, "%f,", &data[i][j]) != 1) {
                fclose(file);
                DELETE_DATA_LABELS(data, num_samples);
                return NULL;
            }
        }
        fgetc(file);
    }

    fclose(file);
    return data;
}


float** read_csv_labels(const char *filename, int num_samples, int num_classes) {
    CHECK_LOAD_ARGS(filename, num_samples, num_classes);

    float **labels = (float **)malloc(num_samples * sizeof(float *));
    for (int i = 0; i < num_samples; ++i) {
        labels[i] = (float *)malloc(num_classes * sizeof(float));
    }

    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file '%s'.\n", filename);
        DELETE_DATA_LABELS(labels, num_samples);
        return NULL;
    }

    for (int i = 0; i < num_samples; ++i) {
        int label;
        if (fscanf(file, "%d,", &label) != 1) {
            fclose(file);
            DELETE_DATA_LABELS(labels, num_samples);
            return NULL;
        }

        for (int j = 0; j < num_classes; ++j) {
            labels[i][j] = (j == label) ? 1.0f : 0.0f;
        }
    }

    fclose(file);
    return labels;
}


void delete_data(float **data, int num_samples) {
    DELETE_DATA_LABELS(data, num_samples);
}


void delete_labels(float **labels, int num_samples) {
    DELETE_DATA_LABELS(labels, num_samples);
}


int split_data(float **data, float **labels, int num_samples, int input_size, int num_classes,
    float test_size, float ***train_data, float ***test_data,
    float ***train_labels, float ***test_labels,
    int *train_count, int *test_count
) {
    if (!data || !labels || num_samples <= 0 || input_size <= 0 || num_classes <= 0 || test_size < 0.0f || test_size > 1.0f) {
        fprintf(stderr, "Error: Invalid input parameters for data splitting.\n");
        return 1;
    }

    *test_count = (int)(num_samples * test_size);
    *train_count = num_samples - *test_count;

    *train_data = malloc(*train_count * sizeof(float *));
    *test_data = malloc(*test_count * sizeof(float *));
    *train_labels = malloc(*train_count * sizeof(float *));
    *test_labels = malloc(*test_count * sizeof(float *));

    if (!*train_data || !*test_data || !*train_labels || !*test_labels) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < *train_count; ++i) {
        (*train_data)[i] = malloc(input_size * sizeof(float));
        (*train_labels)[i] = malloc(num_classes * sizeof(float));

        if (!(*train_data)[i] || !(*train_labels)[i]) {
            fprintf(stderr, "Error: Memory allocation failed for training data or labels.\n");
            return 1;
        }
    }

    for (int i = 0; i < *test_count; ++i) {
        (*test_data)[i] = malloc(input_size * sizeof(float));
        (*test_labels)[i] = malloc(num_classes * sizeof(float));

        if (!(*test_data)[i] || !(*test_labels)[i]) {
            fprintf(stderr, "Error: Memory allocation failed for test data or labels.\n");
            return 1;
        }
    }

    for (int i = 0; i < num_samples; ++i) {
        if (i < *train_count) {
            for (int j = 0; j < input_size; ++j) {
                (*train_data)[i][j] = data[i][j];
            }
            for (int j = 0; j < num_classes; ++j) {
                (*train_labels)[i][j] = labels[i][j];
            }
        } else {
            int test_index = i - *train_count;
            for (int j = 0; j < input_size; ++j) {
                (*test_data)[test_index][j] = data[i][j];
            }
            for (int j = 0; j < num_classes; ++j) {
                (*test_labels)[test_index][j] = labels[i][j];
            }
        }
    }

    return 0;
}


void delete_split_data(float **train_data, float **train_labels,
    float **test_data, float **test_labels,
    int num_samples, int train_count
) {
    if (!train_data || !train_labels || !test_data || !test_labels) return;

    for (int i = 0; i < num_samples; ++i) {
        if (i < train_count) {
            free(train_data[i]);
            free(train_labels[i]);
        } else {
            int test_index = i - train_count;
            free(test_data[test_index]);
            free(test_labels[test_index]);
        }
    }

    free(train_data);
    free(train_labels);
    free(test_data);
    free(test_labels);
}
