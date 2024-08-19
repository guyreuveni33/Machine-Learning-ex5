# Machine Learning Exercise 5: Neural Networks with NumPy and PyTorch

**Author:** Guy Reuveni

**Course:** Machine Learning

## Overview

This project, completed as part of a Machine Learning course, focuses on implementing and training neural networks using both **NumPy** and **PyTorch**. The exercise is divided into two main parts:

1. **Neural Network using NumPy:** Implementation of a multi-layer perceptron (MLP) from scratch for handwriting recognition using the MNIST dataset.
2. **Neural Network using PyTorch:** Implementation of both a fully connected neural network and a convolutional neural network (CNN) for fashion classification using the Fashion-MNIST dataset.

## Part 1: Neural Network using NumPy

### Description

In this section, a neural network is implemented from scratch using only the `numpy` library. The task involves training the network on the MNIST dataset to recognize handwritten digits.

### Steps:

1. **Data Preparation:** The MNIST dataset is loaded, normalized using Min-Max scaling, and split into training and test sets.
2. **Model Implementation:** A multi-layer perceptron (MLP) is implemented with one hidden layer and a softmax output layer.
3. **Training:** The network is trained using backpropagation, with cross-entropy loss calculated at each iteration.
4. **Testing:** The accuracy of the model is evaluated on the test set.

### Key Functions:

- `sigmoid(z)`: Implements the sigmoid activation function.
- `softmax(z)`: Implements the softmax function for the output layer.
- `nll_loss(y_pred, y_true)`: Calculates the negative log likelihood loss.
- `MultilayerPerceptron`: A class that implements the MLP with methods for forward pass, backward pass (backpropagation), training, and testing.

### Results

The trained model achieves an accuracy of 97.19% on the MNIST test set, demonstrating the effectiveness of the neural network implemented using NumPy.

## Part 2: Neural Network using PyTorch

### Description

This section explores the use of PyTorch for building and training neural networks. Two models are implemented: a simple fully connected neural network and a convolutional neural network (CNN), both trained on the Fashion-MNIST dataset.

### Steps:

1. **Data Preparation:** The Fashion-MNIST dataset is loaded, transformed, and split into training and validation sets.
2. **Neural Network Implementation:**
   - **Fully Connected Neural Network (FCNN):** A simple neural network with two hidden layers, implemented and trained to classify fashion items.
   - **Convolutional Neural Network (CNN):** A more complex model that includes convolutional layers, pooling, batch normalization, and dropout to improve performance.
3. **Training and Evaluation:**
   - Both models are trained on the training set and evaluated on the validation set.
   - The CNN model is further fine-tuned to achieve better accuracy on the validation set.
4. **Test Set Predictions:** The best CNN model is used to generate predictions on a test set, which are saved to a file for submission.

### Key Classes:

- `NeuralNetwork`: Implements the fully connected neural network using PyTorch.
- `ConvolutionalNet`: Implements the convolutional neural network (CNN) using PyTorch.
- `ImprovedConvolutionalNet`: An enhanced version of the CNN model with additional layers and regularization techniques.

### Results

The CNN model achieves a validation accuracy of 91.71%, outperforming the simpler fully connected network. The model is trained on the Fashion-MNIST dataset and predictions are saved for the final test set.

## Conclusion

This exercise provides hands-on experience in implementing neural networks from scratch and using advanced frameworks like PyTorch. The project demonstrates the significant improvement in performance when using convolutional layers and other advanced techniques in a CNN.
