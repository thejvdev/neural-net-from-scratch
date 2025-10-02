#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <synapse.h>

// Data parameteres
#define NUM_SAMPLES 60000
#define INPUT_SIZE 28 * 28
#define NUM_CLASSES 10

// Hyperparameteres
#define NUM_LAYERS 3
#define LEARNING_RATE 0.001f
#define BATCH_SIZE 64
#define NUM_EPOCHS 100

#define SAVE_PATH_MAX 64


int main() {
    srand((unsigned int)time(NULL));
    
    // === Data preparation ===
    fputs("Preparing data...\n", stdout);  // Log

    float **data = read_csv_data("../mnist_preparation/data.csv", NUM_SAMPLES, INPUT_SIZE);
    if (!data) {
        fprintf(stderr, "Error: Failed to read data.\n");
        return 1;
    }

    float **labels = read_csv_labels("../mnist_preparation/labels.csv", NUM_SAMPLES, NUM_CLASSES);
    if (!labels) {
        fprintf(stderr, "Error: Failed to read labels.\n");
        return 1;
    }
    
    // Split data
    float **train_data, **test_data, **train_labels, **test_labels;
    int train_count, test_count;

    split_data(data, labels, NUM_SAMPLES, INPUT_SIZE, NUM_CLASSES, 0.2f,
        &train_data, &test_data, &train_labels, &test_labels,
        &train_count, &test_count);
    
    // Memory deallocation
    delete_data(data, NUM_SAMPLES);
    delete_labels(labels, NUM_SAMPLES);

    fputs("Data preparation complete.\n", stdout);  // Log
    
    // === Define model ===
    create_neural_network(NUM_LAYERS);

    init_layer(INPUT_SIZE, 16, relu);
    init_layer(16, 16, relu);
    init_layer(16, NUM_CLASSES, softmax);
    
    // Loss function and optimizer
    setup_loss_function(categorical_cross_entropy);
    setup_optimizer(adam, LEARNING_RATE);
    
    // === Training loop ===
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;

        for (int sample = 0; sample < train_count; ++sample) {
            // Forward pass
            forward(train_data[sample]);
            epoch_loss += compute_loss(train_labels[sample]);

            // Backward pass
            backward(train_data[sample], train_labels[sample]);
            if (sample % BATCH_SIZE == 0 || sample == train_count - 1) {
                update_weights();
                zero_grads();
            }
        }

        if ((epoch + 1) % 10 == 0) {
            printf("Epoch: %d Loss: %f\n", epoch + 1, epoch_loss / train_count);
        }
    }

    // Testing loop
    int num_correct = 0;
    for (int sample = 0; sample < test_count; ++sample) {
        float *predicts = forward(test_data[sample]);
        num_correct += (find_max_index(predicts, NUM_CLASSES) == find_max_index(test_labels[sample], NUM_CLASSES));
    }
    
    float accuracy = (float)num_correct / test_count * 100;
    printf("Accuracy: %.1f%%\n", accuracy);
    
    // Save the model if accuracy >= 93%
    if (accuracy >= 93.0f) {
        char save_path[SAVE_PATH_MAX];
        snprintf(save_path, SAVE_PATH_MAX, "../models/mnist_model_%.1f.bin", accuracy);
        save_neural_network(save_path);
        fputs("Successfully saved.\n", stdout);  // Log
    }
    
    // Memory deallocation
    delete_split_data(train_data, train_labels, test_data, test_labels, NUM_SAMPLES, train_count);
    delete_neural_network();

    return 0;
}
