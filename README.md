Exercise 3 - Multilayer Neural Network Implementation
This repository contains the code for implementing and extending a multilayer artificial neural network (ANN), as described in Chapter 11 of "Machine Learning with PyTorch and Scikit-Learn" by Raschka et al. (2022). The original code, which provides a foundation for this work, can be found at:
- https://github.com/rasbt/machine-learning-book/blob/main/ch11/ch11.ipynb

This exercise extends the original single hidden layer neural network to a two hidden layer architectures. The implementation can be found in the following notebooks:
- Part 2 - https://github.com/edenkor150/Exercise-3/blob/9abc72e754310a9589818bec5ce825cefe1740d4/part2.ipynb
- Part 3 - https://github.com/edenkor150/Exercise-3/blob/9abc72e754310a9589818bec5ce825cefe1740d4/part3.ipynb
- Part 4 - https://github.com/edenkor150/Exercise-3/blob/9abc72e754310a9589818bec5ce825cefe1740d4/part3.ipynb

Project Structure
The repository is structured as follows:
- ch11.ipynb: The original Jupyter Notebook containing the single hidden layer MLP implementation.
- part2.ipynb: A modified Jupyter Notebook extending the original code to include two hidden layers.
- part3.ipynb: Jupyter Notebook for classifying the MNIST dataset with the two hidden layer ANN structure depicted in class.
- part4.ipynb: Jupyter Notebook for implementing the depicted ANN structure using PyTorch.
- README.md: This file, providing an overview of the project and its structure, and links to all ipynb files in the repository.

Summary of Changes
The main changes made in this exercise include:
- Extended the NeuralNetMLP class to accommodate two hidden layers. This involved modifying the __init__, forward, and backward methods of the class.
- Applied the implemented two hidden layer network for classifying the MNIST dataset.
- Compared the predictive performance of the two hidden layer network with the original single hidden layer code from chapter 11, and with a fully connected ANN implemented using PyTorch.
Additional Information
- The MNIST dataset is used for training and evaluating the neural networks. It is loaded using scikit-learn's fetch_openml function.
- The backpropagation algorithm is used for training the neural networks. The code includes implementations of forward and backward propagation.
- The neural network implementation uses a sigmoid activation function.
- The mean squared error (MSE) is used as the loss function.
- Stochastic Gradient Descent (SGD) with mini-batches is used for optimization.
- The code includes helper functions for data loading and evaluation (e.g., minibatch_generator, mse_loss, accuracy).
- You can find additional resources for neural networks and backpropagation at.
References
- Raschka, S., Liu, Y., & Mirjalili, V. (2022). Machine Learning with PyTorch and Scikit-Learn. Packt Publishing.
- The original code is from https://github.com/rasbt/machine-learning-book.
How to Run the Code
1. Clone the repository.
2. Open and run the Jupyter Notebooks (.ipynb files) in the order specified in the structure section.
3. Ensure you have the required libraries installed (numpy, matplotlib, scikit-learn).
This README file is designed to provide a comprehensive overview of the project, making it easy for you to understand the code and reproduce the results. Please let me know if you have any further questions or need any additional assistance.
