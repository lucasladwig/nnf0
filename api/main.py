import numpy as np

class Rede:
    def __init__(self, dict_hiperparameters):
        self.network_matrix = np.empty(0, dtype=object)
        self.activation_functions_dict = {
            "linear": self.linearActFunction,
            "sigmoid": self.sigmoidActFunction,
            "tanh": self.tanhActFunction,
            "relu": self.reluActFunction,
            "leaky_relu": self.leakyReluActFunction,
            "parametric_relu": self.parametricReluActFuntion,
            "elu": self.eluActFunction,
            "swish": self.swishActFunction,
            "softmax": self.softmaxActFunction
        }

    def linearActFunction(self):
        return
    
    def derivativeLinear(self):
        return

    def sigmoidActFunction(self):
        return
    
    def derivativeSigmoid(self):
        return

    def tanhActFunction(self):
        return

    def derivativeTanh(self):
        return

    def reluActFunction(self):
        return
    
    def derivativeRelu(self):
        return

    def leakyReluActFunction(self):    
        return
    
    def derivativeLeakyRelu(self):
        return
    
    def parametricReluActFuntion(self):
        return
    
    def derivativeParametricRelu(self):
        return
    
    def eluActFunction(self):
        return
    
    def derivativeELU(self):
        return
    
    def swishActFunction(self, x_input_value):
        y_output_value = x_input_value * (1 / (1 + np.exp(-x_input_value)))
        return y_output_value
    
    def derivativeSwish(self):
        return
    
    def softmaxActFunction(self, x_input_vector):
        exp_values_vector = np.exp(x_input_vector)
        exp_values_vector = exp_values_vector / np.sum(exp_values_vector)
        return exp_values_vector
    
'''
    eu fiz a derivada da softmax, 
    mas pelo que li na web essa função n importa muito 
    pois já está incorporada no cálculo da cross entropy categorica

    def derivativeSoftmax(self, softmax_vector): 
        sigma = np.array(softmax_vector)
        jacobian_matrix = np.diag(sigma) - np.outer(sigma, sigma)
        return jacobian_matrix
'''

    def squareErrorCostFunction(self):
        return
    
    def crossEntropyCostFunction(self):
        return
    
    def categoricCrossEntropyCostFunction(self, y_predicted_vector, y_true_vector):
        loss_value = -np.sum(y_true_vector * np.log(y_predicted_vector))
        return loss_value
    
    def feedFoward(self):
        return

    def backPropagation(self):
        return
    
    def gradientDescent(self):
        return
    
    def linearCombinerNeuron(self, k_number_neurons_in_layer, w_weight_vector, x_inputs_vector):
        v_sum_value = 0
        for i in range(k_number_neurons_in_layer):
            v_sum_value  += w_weight_vector[i]*x_inputs_vector[i]
        return v_sum_value 
    
    '''
    Aqui a gente está passando o nome da função a ser chamada (ex. "relu"), que será usada como chave do dicionário de funções
    que armazena uma referência ao método correspondente. Então quando a estrutura de dicionário chamar essa referência, os parênteses
    do lado que armazenam o valor de input serão passados para o método 'invocado'.

    ex:
    no código ficará                --> self.activation_functions_dict["relu"](-0.5) 
    mas isso vai se transformar em  --> self.reluActFunction(x)
    e retornará                     --> 0
    '''
    def useActFunction(self, function_name_string, v_sum_value):
        y_output_value = self.activation_functions_dict[function_name_string](v_sum_value)
        return y_output_value
    
    def neuronIteration(self, k_number_neurons_in_layer, w_weight_vector, x_inputs_vector, function_name_string):
        v_sum_value = self.linearCombinerNeuron(k_number_neurons_in_layer, w_weight_vector, x_inputs_vector)
        y_output_value = self.useActFunction(function_name_string, v_sum_value)
        return y_output_value