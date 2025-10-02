# Neural Network in the C Language

This repository was created to deepen my understanding of neural networks, enhance my programming skills, and gain experience in building libraries in C.

## Description

This project implements a library named **Synapse** for constructing various MLP (Multilayer Perceptron) architectures in C/C++. The library is optimized for fast training of large models and offers flexibility for easily adjusting hyperparameters. By adhering to encapsulation within the project’s modules, the library simplifies usage while maintaining full control over the training process.

## Installation

To clone this repository, execute the following bash command:

```bash
git clone https://github.com/profjuvii/neural-net-from-scratch.git
```

## Navigation

The repository is organized into several folders, each serving a specific purpose:

- **datasets**: Stores raw dataset archives before preprocessing. These files are used by the preparation scripts to generate CSV inputs for training.
- **mnist_preparation**: Includes a Python script for preparing handwritten digit data from the MNIST dataset in CSV format.
- **mnist_training**: Contains code for training a model on pre-processed MNIST data.
- **models**: Stores pre-trained models.
- **synapse**: Contains the core neural network library.

To train your own model, ensure that the necessary compiler extensions are included as follows:

```bash
-I../synapse/include -L../synapse/lib -lsynapse
```

> [!NOTE]
>
> This depends on the location of the project files in your environment.

## Requirements for Training the MNIST Model

To run the code for training a neural network on handwritten MNIST data, follow these steps:

1. **Prepare MNIST Data**:
   - In the `mnist_preparation/` folder, you need set up a Python virtual environment:
     ```bash
     python -m venv .venv
     ```
   - Activate the virtual environment:
     ```bash
     source .venv/bin/activate
     ```
   - Install the necessary library:
     ```bash
     pip install idx2numpy
     ```
   - After this, you can run the MNIST data preparation script:
     ```bash
     python mnist2csv.py
     ```

2. **Build and Run the Training Code**:
   - In the `mnist_training/` folder, run the following command to compile the code:
     ```bash
     make
     ```
   - Once the build is complete, execute the program using:
     ```bash
     ./train
     ```

Once these steps are completed, the training of the neural network on the MNIST dataset will begin as intended.

Thanks to the hyperparameter settings defined in the code, I achieved an accuracy of 94.2%. The trained model has been saved in the `models/` folder.

> [!NOTE]
>
> You can also find all the library functions in the header files located in `synapse/include/`.

## Author

Created by [Denys Bondarchuk](https://github.com/profjuvii). Feel free to reach out or contribute to the project!
