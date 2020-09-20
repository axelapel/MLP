import numpy as np


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
        output = input_vector
        return output

    def backward(self, estimation, training_y):
        """Backward propagation to correct the weights.

        Parameters
        ----------
        estimation : 1D array
            Output after feedforward.
        training_y : 1D array
            Training output.
        """
        error = (estimation - training_y)
        for i in reversed(range(self.number_layers + 1)):
            delta = self.sig_der(self.hidden[i]).dot(error)
            grad = self.activations[i].dot(delta)
            self.gradients.append(grad)
            error = (delta * self.weights[i].T)[0]
        self.gradients.reverse()

    def grad_descent(self, learning_rate):
        """Gradient descent to correct the weights by minimizing the
        loss function.

        Parameters
        ----------
        learning_rate : float
            0 < lr < 1.
        """
        for i in range(self.number_layers + 1):
            self.weights[i] -= learning_rate * self.gradients[i]

    def train(self, input, training_output, learning_rate):
        """Method to implement in a loop in order to train the neural network.

        Parameters
        ----------
        training_input : 1D array
            Initial x-data.
        training_output : 1D array
            Corresponding y-data.
        learning_rate : float
            0 < lr < 1.
        """
        y_hat = self.forward(input)
        self.backward(y_hat, training_output)
        self.grad_descent(learning_rate)

    def get_weights(self):
        """ Getter method to keep track of weights arrays."""
        return self.weights


if __name__ == "__main__":

    # Hyper-parameters
    n_in = 2
    n_out = 1
    n_lay = 1
    sz_hid = 3

    # Training I/O
    x = np.random.rand(n_in)
    y = np.random.rand(n_out)

    # NN instance
    MLP = MLP(n_lay, sz_hid, n_in, n_out)

    #y_hat = MLP.forward(x)
    #print("Output : {}".format(y_hat))
    # W = MLP.get_weights()
    # print(W[0])

    epochs = 10
    for i in range(epochs):
        MLP.train(x, y, 1e-2)
        w1 = MLP.get_weights()[0]
        print(w1)
