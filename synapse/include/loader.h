#ifndef LOADER_H
#define LOADER_H

float** read_csv_data(const char *filename, int num_samples, int input_size);
float** read_csv_labels(const char *filename, int num_samples, int num_classes);

void delete_data(float **data, int num_samples);
void delete_labels(float **labels, int num_samples);

int split_data(float **data, float **labels, int num_samples, int input_size, int num_classes,
    float test_size, float ***train_data, float ***test_data,
    float ***train_labels, float ***test_labels,
    int *train_count, int *test_count
);

void delete_split_data(float **train_data, float **train_labels,
    float **test_data, float **test_labels,
    int num_samples, int train_count
);

#endif
