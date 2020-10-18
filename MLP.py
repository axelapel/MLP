import numpy as np


class MLP(object):
    """
    Fully connected neural network with arbitrary number of layers,
    input, output and hidden units.
    """

    def __init__(self, size_input, hidden_layers, size_output):

        self.size_input = size_input
        self.hidden_layers = hidden_layers
        self.size_output = size_output
        self.total_loss = []

        layers = [size_input] + hidden_layers + [size_output]

        # Random initialization of weights and biases
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        gradients = []
        for i in range(len(layers) - 1):
            der = np.zeros((layers[i], layers[i + 1]))
            gradients.append(der)
        self.gradients = gradients

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def _sigmoid(self, arg):
        """Sigmoid activation function.

        Parameters
        ----------
        arg : float/array
        """
        return 1. / (1. + np.exp(-arg))

    def _sigmoid_derivative(self, sigmoid):
        """Derivative of the sigmoid activation function

        Parameters
        ----------
        sigmoid : float/array
            Sigmoid function of a given float/array.
        """
        return sigmoid * (1. - sigmoid)

    def _mse_loss(self, y_hat, y):
        """Loss function : mean squared error.

        Parameters
        ----------
        y_hat : array
            Estimated output.
        y : array
            Training output.
        """
        return 0.5 * (y_hat - y)**2

    def forward(self, inputs):
        """Forward propagation of the input through the network.

        Parameters
        ----------
        inputs : array
            Input data.

        Returns
        -------
        Output without details about the weights.
        """
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            hidden = np.dot(self.activations[i], w)
            self.activations[i+1] = self._sigmoid(hidden)

        return self.activations[-1]

    def backward(self, error):
        """Backward propagation to correct the weights/biases.

        Parameters
        ----------
        error : 1D array
            Residuals.
        """
        for i in reversed(range(len(self.gradients))):
            delta = error * self._sigmoid_derivative(self.activations[i+1])
            self.gradients[i] = np.dot(
                self.activations[i].reshape(-1, 1), delta.reshape(-1, 1).T)
            error = np.dot(delta, self.weights[i].T)

    def gradient_descent(self, learning_rate):
        """Gradient descent to correct the weights by minimizing the
        loss function.

        Parameters
        ----------
        learning_rate : float
            0 < lr < 1.
        """
        for i in range(len(self.weights)):
            weights = self.weights[i]
            weights -= self.gradients[i] * learning_rate
            self.weights[i] = weights

    def train(self, inputs, targets, epochs, learning_rate):
        """Training method.

        Parameters
        ----------
        inputs : 1D array
        targets : 1D array
            Training output
        epochs : float
            Total number of iteration for the training.
        learning_rate : float
            0 < lr < 1
        """
        for i in range(epochs):
            sum_errors = 0
            for j, input in enumerate(inputs):
                target = targets[j]
                output = self.forward(input)
                error = output - target
                self.backward(error)
                self.gradient_descent(learning_rate)
                sum_errors += self._mse_loss(target, output)

            print("\r[epoch {}] : loss = {}".format(
                i+1, sum_errors / len(inputs)), end="", flush=True)
            self.total_loss.append(sum_errors / len(inputs))


if __name__ == "__main__":

    """
    Test with a simple regression for the addition of 2 numbers
    """

    import matplotlib.pyplot as plt

    # Parameters
    N = 1000
    sz_in = 2
    sz_out = 1
    hidden = 3
    lr = 1e-1
    epochs = 100

    # Trainsets
    train_x = np.random.uniform(low=0.0, high=0.5, size=(N, 2))
    train_y = np.sum(train_x, axis=1).reshape(-1, 1)

    # Network instance
    mlp = MLP(sz_in, [hidden], sz_out)
    mlp.train(train_x, train_y, epochs, lr)

    # Test
    input = np.array([0.3, 0.1])
    target = np.array([0.4])
    output = mlp.forward(input)

    print("\n\nTEST:")
    print("{} + {} = {}".format(input[0], input[1], output[0]))

    fig, ax = plt.subplots()
    ax.plot(np.arange(epochs)+1, mlp.total_loss, "k")
    ax.set(xlabel="Epochs", ylabel="Total loss")
    ax.grid()
    plt.savefig("training_MLP.png")
