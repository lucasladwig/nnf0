import numpy as np

class Rede:
    def __init__(self, dict_hiperparameters):
        self.network_matrix = np.empty(0, dtype=object)
        self.activation_functions = {
            "binary_step": self.binaryStepActFunction,
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
    
    def swishActFunction(self):
        return
    
    def derivativeSwish(self):
        return
    
    def softmaxActFunction(self):
        return
    
    def derivativeSoftmax(self):
        return
    
    def squareErrorCostFunction(self):
        return
    
    def crossEntropyCostFunction(self):
        return
    
    def categoricCrossEntropyCostFunction(self):
        return
    
    def feedFoward(self):
        return

    def backPropagation(self):
        return
    
    def gradientDescent(self):
        return
    
    def linearCombinerNeuron(self, x_inputs_vector, w_weight_matrix):
        return
    
    def useActFunction(self, act_function_name) # a ideia é que receba o nome da função como string, que será usada como key do dicionário. Python permite invocar métodos deste jeito
        # y = self.activation_functions[act_function_name](v)
        return