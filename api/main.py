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
             "parametric_relu_x": self.param_relu_deriv_x,
             "parametric_relu_a": self.param_relu_deriv_a,
             "elu": self.elu_deriv,
             "swish": self.swish_deriv,
             "softmax": self.softmax_deriv
        }
        self.cost_functions = {
            "mean_squared_error": self.squared_error,
            "binary_cross_entropy": self.binary_cross_entropy,
            "categoric_cross_entropy": self.categoric_cross_entropy
        }
        self.vector_activation_functions = {"softmax"} # ativações aplicadas à camada inteira de uma vez, não neurônio a neurônio
        self.leaky_relu_apha = 0.01 # valor padrão para a leaky relu. Basta mudar o valor do atributo.
        self.elu_alpha = 1.0 # valor padrão para a elu. Basta mudar o valor do atributo.
        self.chosen_cost_function = None
        self.layers_activation_func_list = []
        self.layer_inputs = [] # cache: vetor de entrada de cada camada (com bias), preenchido no feedforward e usado na backpropagation
        self.layer_z = [] # cache: combinações lineares (pré-ativações) de cada camada, usadas na backpropagation
        self.param_relu_alphas = [] # alpha (a) aprendível por neurônio de cada camada parametric_relu; None nas demais camadas
        self.param_relu_alpha_gradients = [] # gradiente do alpha por camada, preenchido na backpropagation para o gradient_descent usar
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
    def tanh_activ(self, x_input_value):
        return (np.exp(x_input_value) - np.exp(-x_input_value)) / (np.exp(x_input_value) + np.exp(-x_input_value))

    def tanh_deriv(self, x_input_value):
        t = self.tanh_activ(x_input_value)
        return 1 - t ** 2

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
    def param_relu_activ(self, x_input_vector, alpha_vector):
        """Parametric ReLU activation function (vectorized over a layer).
            Params:
            x_input_vector: pre-activations of the layer's neurons
            alpha_vector: per-neuron learned slope for negative values
        """
        return np.where(x_input_vector > 0, x_input_vector, alpha_vector * x_input_vector)

    def param_relu_deriv_x(self, x_input_vector, alpha_vector):
        """Parametric ReLU derivative with regards to input 'x' (vectorized).
            Params:
            x_input_vector: pre-activations of the layer's neurons
            alpha_vector: per-neuron learned slope for negative values
        """
        return np.where(x_input_vector > 0, 1.0, alpha_vector)

    def param_relu_deriv_a(self, x_input_vector):
        """Parametric ReLU derivative with regards to learned parameter 'a' (vectorized).
            Params:
            x_input_vector: pre-activations of the layer's neurons
        """
        return np.where(x_input_vector > 0, 0.0, x_input_vector)

    # --- ELU ---
    def elu_activ(self, x_input_value):
        return np.where(x_input_value > 0, x_input_value, self.elu_alpha * (np.exp(x_input_value) - 1))

    def elu_deriv(self, x_input_value):
        return np.where(x_input_value > 0, 1.0, self.elu_alpha * np.exp(x_input_value))

    # --- Swish ---
    def swish_activ(self, x_input_value):
        y_output_value = x_input_value * (1 / (1 + np.exp(-x_input_value)))
        return y_output_value

    def swish_deriv(self, x_input_value):
        sigmoid_value = self.sigmoid_activ(x_input_value)
        return sigmoid_value + x_input_value * sigmoid_value * (1 - sigmoid_value)

    # --- Softmax ---
    def softmax_activ(self, x_input_vector):
        # subtrai o máximo para estabilidade numérica (evita overflow no exp); por ser invariante a deslocamento, não altera o resultado
        exp_values_vector = np.exp(x_input_vector - np.max(x_input_vector))
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
    def back_propagation(self, y_predicted_vector, y_true_vector):
        # Calcula o gradiente da perda em relação aos pesos de cada camada.
        # Usa o cache preenchido pelo feedforward: self.layer_inputs e self.layer_z.
        # Retorna uma lista de matrizes de gradiente, uma por camada (mesmo formato de self.network).
        # Também preenche self.param_relu_alpha_gradients com o gradiente do alpha das camadas parametric_relu.
        num_layers = len(self.network)
        gradients = [None] * num_layers
        self.param_relu_alpha_gradients = [None] * num_layers
        last_index = num_layers - 1

        # --- erro (delta) da camada de saída ---
        # upstream = dL/d(saída da camada): o gradiente que chega na saída, antes de multiplicar por g'(z)
        if self.chosen_cost_function == self.squared_error:
            # erro quadrático: dL/dy = 2*(y_pred - y_true)
            upstream = 2 * (y_predicted_vector - y_true_vector)
            delta = upstream * self.calc_activation_deriv(last_index)
        else:
            # entropia cruzada com sigmoide/softmax: o delta da saída simplifica para (y_pred - y_true)
            upstream = None
            delta = y_predicted_vector - y_true_vector

        # --- percorre as camadas de trás para frente ---
        for layer_index in reversed(range(num_layers)):
            # gradiente desta camada = produto externo entre o delta e a entrada da camada (já com bias)
            gradients[layer_index] = np.outer(delta, self.layer_inputs[layer_index])
            # gradiente do alpha, se esta for uma camada parametric_relu (upstream = dL/d(saída) desta camada)
            if self.layers_activation_func_list[layer_index] == "parametric_relu" and upstream is not None:
                self.param_relu_alpha_gradients[layer_index] = upstream * self.param_relu_deriv_a(self.layer_z[layer_index])
            if layer_index > 0:
                # propaga o erro para a camada anterior
                propagated_error = self.network[layer_index].T.dot(delta) # tamanho = nº de entradas + bias
                upstream = propagated_error[:-1] # descarta a linha do bias; é o dL/d(saída) da camada anterior
                delta = upstream * self.calc_activation_deriv(layer_index - 1) # multiplica pela inclinação da ativação

        return gradients

    def calc_activation_deriv(self, layer_index):
        """Retorna g'(z) da camada, tratando a parametric_relu à parte porque sua
        derivada depende do alpha (a) de cada neurônio, e não apenas da pré-ativação z.
        """
        func_name = self.layers_activation_func_list[layer_index]
        z_vector = self.layer_z[layer_index]
        if func_name == "parametric_relu":
            return self.param_relu_deriv_x(z_vector, self.param_relu_alphas[layer_index])
        deriv_func = self.activation_functions_deriv[func_name]
        return deriv_func(z_vector)

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

    def set_cost_function(self, cost_function_name: str):
        self.chosen_cost_function = self.cost_functions[cost_function_name]

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
        # parametric_relu: cada neurônio recebe seu próprio alpha (a) aprendível, iniciado em 0.25 (padrão da PReLU)
        if func_name == "parametric_relu":
            self.param_relu_alphas.append(np.full(num_neurons, 0.25))
        else:
            self.param_relu_alphas.append(None)
    
    def calc_layer_output(self, layer_index: int, input_vector: np.array):
        layer = self.network[layer_index] #pega a matriz de pesos da camada em questão
        func_name = self.layers_activation_func_list[layer_index]
        z_vector = [] # combinações lineares (pré-ativações) de cada neurônio, necessárias na backpropagation
        for neuron_index in range(layer.shape[0]):
            linear_combination_neuron = (layer[neuron_index].dot(input_vector)) # multiplicação da linha de pesos do neurônio pelo vetor de entrada, resultando na combinação linear dos inputs para aquele neurônio
            z_vector.append(linear_combination_neuron) # guarda a pré-ativação antes de aplicar a função de ativação
        z_vector = np.array(z_vector)
        if func_name == "parametric_relu":
            # parametric_relu depende do alpha (a) de cada neurônio: recebe o vetor de pré-ativações e os alphas da camada
            output_vector = self.param_relu_activ(z_vector, self.param_relu_alphas[layer_index])
        elif func_name in self.vector_activation_functions:
            # ativações que dependem da camada inteira (ex.: softmax) recebem o vetor de pré-ativações completo
            output_vector = self.activation_functions[func_name](z_vector)
        else:
            # ativações elemento a elemento: aplica a função de ativação a cada neurônio separadamente
            output_vector = []
            for z_value in z_vector:
                output_vector.append(self.activate_neuron(func_name, z_value)) # aplica a função de ativação à pré-ativação, resultando na saída do neurônio
            output_vector = np.array(output_vector)
        return output_vector, z_vector
    
    # === FEED FORWARD ===
    def feedforward(self, input_vector, y_true_vector):
        self.layer_inputs = [] # zera o cache a cada passagem para frente
        self.layer_z = []
        current_input = input_vector
        last_layer_index = len(self.network) - 1
        for layer_index in range(len(self.network)):
            self.layer_inputs.append(current_input) # guarda a entrada desta camada (já com bias) para a backpropagation
            output_vector, z_vector = self.calc_layer_output(layer_index, current_input)
            self.layer_z.append(z_vector) # guarda as pré-ativações desta camada
            if layer_index < last_layer_index:
                current_input = np.append(output_vector, 1.0) # adiciona o bias apenas entre camadas, como entrada da próxima
            else:
                current_input = output_vector # camada de saída: a predição não recebe bias
        prediction = current_input
        loss = self.get_loss(prediction, y_true_vector)
        return prediction, loss
     
    def get_loss(self, y_predicted_vector, y_true_vector):
        loss_value = self.chosen_cost_function(y_predicted_vector, y_true_vector)
        return loss_value