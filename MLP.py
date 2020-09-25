import numpy as np
import matplotlib.pyplot as plt
import pdb


class MLP:

    def __init__(self, number_layers, size_hidden, size_input, size_output):
        """Fully connected neural network with arbitrary number of layers,
        input, output and hidden units.

        /!\ In the following code, N layers corresponds to N hidden layers
        /!\ (ie. [N+1] connexions).
        """
        self.number_layers = number_layers
        self.size_hidden = size_hidden
        self.size_input = size_input
        self.size_output = size_output

        self.hidden = []
        self.activations = []
        self.gradients = []
        self.bias_errors = []

        # Input layer
        self.weights = [np.random.rand(self.size_input, self.size_hidden)]
        self.biases = [np.random.rand(self.size_hidden)]

        # Hidden layers*
        for i in range(1, self.number_layers):
            self.weights.append(np.random.rand(
                self.size_hidden, self.size_hidden))
            self.biases.append(np.random.rand(self.size_hidden))

        # Output layer
        self.weights.append(np.random.rand(
            self.size_hidden, self.size_output))
        self.biases.append(np.random.rand(self.size_output))

    def sigmoid(self, arg):
        """Sigmoid activation function.

        Parameters
        ----------
        arg : float/array
        """
        sigmoid = 1. / (1. + np.exp(-arg))
        return sigmoid

    def sig_der(self, sigmoid):
        """Derivative of the sigmoid activation function

        Parameters
        ----------
        sigmoid : float/array
            Sigmoid function of a given float/array.
        """
        derivative = sigmoid * (1. - sigmoid)
        return derivative

    def loss(self, y_hat, y):
        """Loss function : mean squared error.

        Parameters
        ----------
        y_hat : array
            Estimated output.
        y : array
            Training output.
        """
        return 0.5 * (y_hat - y)**2

    def forward(self, input_vector):
        """Forward processing of the input through the network.

        Parameters
        ----------
        input_vector : array
            Input data.

        Returns
        -------
        Output without details about the weights.
        """
        self.activations.append(input_vector)
        for i in range(self.number_layers + 1):
            dot = np.dot(input_vector, self.weights[i]) + self.biases[i]
            self.hidden.append(dot)
            input_vector = self.sigmoid(dot)
            self.activations.append(input_vector)
        self.output = input_vector
        return self.output

    def backward(self, estimation, training_y):
        """Backward propagation to correct the weights/biases.

        Parameters
        ----------
        estimation : 1D array
            Output after feedforward.
        training_y : 1D array
            Training output.
        """
        error = (estimation - training_y).reshape(-1, 1)
        for i in reversed(range(self.number_layers + 1)):
            pdb.set_trace()
            delta = np.multiply(error,
                                self.sig_der(self.hidden[i].reshape(-1, 1)))
            self.bias_errors.append(delta[:, 0])
            grad = np.dot(self.activations[i].reshape(-1, 1), delta.T)
            self.gradients.append(grad)
            error = np.dot(self.weights[i], delta)

        # The dimensions are good but the values of gradients/delta are
        # exagerated : see values.

    def grad_descent(self, learning_rate):
        """Gradient descent to correct the weights by minimizing the
        loss function.

        Parameters
        ----------
        learning_rate : float
            0 < lr < 1.
        """
        self.gradients.reverse()
        self.bias_errors.reverse()
        for i in range(self.number_layers + 1):
            # pdb.set_trace()
            self.weights[i] -= learning_rate * self.gradients[i]
            self.biases[i] -= learning_rate * self.bias_errors[i]

        # WEIGHTS and BIASES ARE WAY TOO LARGE !
        # ---> Problem in the backpropagation

    def get_weights(self):
        """ Getter method to keep track of weights arrays."""
        return self.weights


if __name__ == "__main__":

    # Hyper-parameters
    n_in = 3
    n_out = 1
    n_lay = 4
    sz_hid = 10
    lr = 1e-3

    # Training I/O
    train_x = np.random.rand(n_in)
    train_y = np.random.rand(n_out)

    # NN instance
    MLP = MLP(n_lay, sz_hid, n_in, n_out)

    #y_hat = MLP.forward(x)
    #print("Output : {}".format(y_hat))
    # W = MLP.get_weights()
    # print(W[0])

    epochs = np.arange(100)
    loss = np.zeros_like(epochs)
    for epoch in epochs:
        output = MLP.forward(train_x)
        MLP.backward(output, train_y)
        MLP.grad_descent(lr)
        loss[epoch] = MLP.loss(output, train_y)

    fig, ax = plt.subplots()
    ax.plot(epochs, loss)
