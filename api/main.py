import numpy as np


class Rede:
    def __init__(self, learning_rate: float, atributes = None, labels = None):
        self.network = []
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
        self.cost_functions = {
            "mean_squared_error": self.squared_error,
            "binary_cross_entropy": self.binary_cross_entropy,
            "categoric_cross_entropy": self.categoric_cross_entropy
        }
        self.leaky_relu_apha = 0.01 # valor padrão para a leaky relu. Basta mudar o valor do atributo.
        self.layers_activation_func_list = []
        self.weights_initialization_mode = "zeros" # "zeros" por default, mas também admite "random"
        self.atributes = atributes # imagine que aqui tem um dataframe, porém sem a coluna de resposta
        self.labels = labels # imagine que aqui tem um dataframe, porém apenas com a coluna de resposta

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
    def param_relu_activ(self, x: float, a: float) -> float:
        """Parametric ReLU activation function.
            Params:
            x: input vector
            a: linear constant for negative values (learned by the network)
        """
        return x if x > 0 else a * x

    def param_relu_deriv_x(self, x: float, a: float) -> float:
        """Parametric ReLU derivative with regards to input 'x'.
            Params:
            x: input vector
            a: linear constant for negative values (learned by the network)
        """

    def param_relu_deriv_a(self, x: float) -> float:
        """Parametric ReLU derivative with regards to learned parameter 'a'.
            Params:
            x: input vector
            a: linear constant for negative values (learned by the network)
        """
        return 0.0 if x > 0 else x

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
    def squared_error(self, y_predicted_vector, y_true_vector):
        loss_value = (y_true_vector[0] - y_predicted_vector[0]) ** 2
        return loss_value

    def binary_cross_entropy(self, y_predicted_vector, y_true_vector):
        loss_value = -(y_true_vector[0] * np.log(y_predicted_vector[0]) + ((1 - y_true_vector[0]) * np.log(1 - y_predicted_vector[0])))
        return loss_value

    def categoric_cross_entropy(self, y_predicted_vector, y_true_vector):
        loss_value = -np.sum(y_true_vector * np.log(y_predicted_vector))
        return loss_value

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

    # === LAYER LOGIC ===
    def create_initial_layer(self, num_neurons: int, func_name: str):
        quantity_of_inputs = self.atributes.shape[1] + 1 # quantidade de atributos, mais 1 para o bias.
        self.create_layer(num_neurons, func_name, quantity_of_inputs)

    def create_hidden_layer(self, num_neurons: int, func_name: str):
        quantity_of_inputs = len(self.network[-1]) + 1 # quantidade de neurônios da camada anterior, mais 1 para o bias.
        self.create_layer(num_neurons, func_name, quantity_of_inputs)

    def create_output_layer(self, num_neurons: int, func_name: str, cost_func_name: str):
        quantity_of_inputs = len(self.network[-1]) + 1 # quantidade de neurônios da camada anterior, mais 1 para o bias.
        self.create_layer(num_neurons, func_name, quantity_of_inputs)

    def create_layer(self, num_neurons: int, func_name: str, quantity_of_inputs: int):
        w_matrix = [] # inicia como lista comum. Cada linha aqui representa os pesos de um neurônio.
        for neuron in range(num_neurons):
            if self.weights_initialization_mode == "zeros":
                w_vector_aux = [0.0] * quantity_of_inputs
            else:
                w_vector_aux = np.random.rand(quantity_of_inputs).tolist()
            w_matrix.append(w_vector_aux) # lista de listas
        camada = np.array(w_matrix, dtype=float) # transforma a lista de listas em matriz numpy, onde cada linha representa os pesos de um neurônio.
        self.network.append(camada)
        self.layers_activation_func_list.append(func_name)
    
    def calc_layer_output(self, layer_index: int, input_vector: np.array):
        layer = self.network[layer_index] #pega a matriz de pesos da camada em questão
        func_name = self.layers_activation_func_list[layer_index]
        output_vector = []
        for neuron_index in range(layer.shape[0]):
            linear_combination_neuron = (layer[neuron_index].dot(input_vector)) # multiplicação da linha de pesos do neurônio pelo vetor de entrada, resultando na combinação linear dos inputs para aquele neurônio
            output_vector.append(self.activate_neuron(func_name, linear_combination_neuron)) # aplica a função de ativação à combinação linear, resultando na saída do neurônio
        output_vector.append(1.0) # adiciona o valor do bias
        return np.array(output_vector)
    
    # === FEED FORWARD ===
    def feedforward(self, input_vector: np.array):
        print("=== INICIANDO FEEDFORWARD ===")
        output_vector = input_vector
        print("input_vector da rede: ", output_vector)
        for layer_index in range(len(self.network)):
            output_vector = self.calc_layer_output(layer_index, output_vector)
            print("camada ", layer_index, " - output_vector: ", output_vector)
        return output_vector