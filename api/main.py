import numpy as np


class Rede:
    def __init__(self, learning_rate: float):
        self.network = np.empty(0, dtype=object)
        self.learning_rate = learning_rate
        self.activation_functions = {
            "linear": self.linear_activ,
            "sigmoid": self.sigmoid_activ,
            "tanh": self.tanh_activ,
            "relu": self.relu_activ,
            "leaky_relu": self.leaky_relu_activ,
            "parametric_relu": self.param_relu_activ,
            "elu": self.elu_activ,
            "swish": self.swish_activ,
            "softmax": self.softmax_activ
        }
        self.leaky_relu_apha = 0.01 # valor padrão para a leaky relu. Basta mudar o valor do atributo.

    # Teste de dict das funções com suas derivadas direto no valor do dict
        self.activation_functions_deriv = {
             "linear": self.linear_deriv,
             "sigmoid": self.sigmoid_deriv,
             "tanh": self.tanh_deriv,
             "relu": self.relu_deriv,
             "leaky_relu": self.leaky_relu_deriv,
             "parametric_relu": self.param_relu_deriv,
             "elu": self.elu_deriv,
             "swish": self.swish_deriv,
             "softmax": self.softmax_deriv
        }

    # === ACTIVATION FUNCTIONS (AND DERIVATIVES) ===
    # --- Linear ---
    def linear_activ(self, x: float, a: float = 1):
        """Linear activation function.
            Params:
            x: input vector
            a: linear constant
        """
        return a * x

    def linear_deriv(self, a: float = 1):
        """Linear activation function derivative, used for backpropagation.
            Params:
            a: linear constant
        """
        return a

    # --- Sigmoid ---
    def sigmoid_activ(self, x_input_value):
        return 1 / (1 + np.exp(-x_input_value))

    def sigmoid_deriv(self, x_input_value):
        s = self.sigmoid_activ(x_input_value)
        return s * (1 - s)

    # --- TanH ---
    def tanh_activ(self):
        return

    def tanh_deriv(self):
        return

    # --- ReLU ---
    def relu_activ(self, x_input_value):
        return np.maximum(0, x_input_value)

    def relu_deriv(self, x_input_value):
        return np.where(x_input_value > 0, 1.0, 0.0)

    # --- Leaky ReLU ---
    def leaky_relu_activ(self, x_input_value):
        return np.maximum(self.leaky_relu_apha * x_input_value, x_input_value)

    def leaky_relu_deriv(self, x_input_value):
        return np.where(x_input_value > 0, 1.0, self.leaky_relu_apha)

    # --- Parameter ReLU ---
    def param_relu_activ(self):
        return

    def param_relu_deriv(self):
        return

    # --- ELU ---
    def elu_activ(self):
        return

    def elu_deriv(self):
        return

    # --- Swish ---
    def swish_activ(self, x_input_value):
        y_output_value = x_input_value * (1 / (1 + np.exp(-x_input_value)))
        return y_output_value

    def swish_deriv(self):
        return

    # --- Softmax ---
    def softmax_activ(self, x_input_vector):
        exp_values_vector = np.exp(x_input_vector)
        exp_values_vector = exp_values_vector / np.sum(exp_values_vector)
        return exp_values_vector

    def softmax_deriv(self, softmax_vector):
        # talvez não seja necesária
        sigma = np.array(softmax_vector)
        jacobian_matrix = np.diag(sigma) - np.outer(sigma, sigma)
        return jacobian_matrix

    # === COST FUNCTIONS ===
    def mean_squared_error(self):
        return

    def binary_cross_entropy(self):
        return

    def categoric_cross_entropy(self, y_predicted_vector, y_true_vector):
        loss_value = -np.sum(y_true_vector * np.log(y_predicted_vector))
        return loss_value

    # === FEED FORWARD ===
    def feed_forward(self):
        return

    # === BACKPROPAGATION ===
    def back_propagation(self):
        return

    # === GRADIENT DESCENT ===
    def gradient_descent(self):
        return

    # === NEURON LOGIC ===
    def linear_combination(self, k: int, w: np.array, x: np.array):
        """
            Params:
                k: number of neurons in layer
                w: weight vector
                x: input vector
        """
        v_sum = 0
        for i in range(k):
            v_sum += w[i] * x[i]
        return v_sum

    def activate_neuron(self, func_name: str, v_sum: float):
        y_output = self.activation_functions[func_name](v_sum)
        return y_output

    def neuron_iteration(self, k, w, x, func_name):
        """
            Params:
                k: number of neurons in layer
                w: weight vector
                x: input vector
        """
        v_sum = self.linear_combination(k, w, x)
        y_output = self.activate_neuron(func_name, v_sum)
        return y_output
