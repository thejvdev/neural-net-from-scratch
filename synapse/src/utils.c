#include <stdio.h>
#include "utils.h"


void print_vec(const float *vec, int size, int vallen) {
    if (!vec || size <= 0 || vallen > 12 || vallen < 0) return;
    
    printf("[");
    for (int i = 0; i < size - 1; ++i) {
        printf("%.*f, ", vallen, vec[i]);
    }

    printf("%.*f]\n", vallen, vec[size - 1]);
}


int find_max_index(const float *vec, int size) {
    if (!vec || size <= 0) {
        fprintf(stderr, "Error in %s(): Invalid input parameters.\n", __func__);
        return -1;
    }

    int class_index = 0;
    for (int i = 1; i < size; ++i) {
        if (vec[class_index] < vec[i]) class_index = i;
    }

    return class_index;
}
