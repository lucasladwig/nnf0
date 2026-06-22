from api.main import Rede
import numpy as np

print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
# imagine que aqui tem um dataframe, porém sem a coluna de resposta
data = [[0.5, 0.2, 0.1], [0.9, 0.8, 0.7]]
# converti para np.array porque no código ele pega o tamanho por shape()
data_vector = np.array(data)
# imagine que aqui tem um dataframe, porém apenas com a coluna de resposta
labels = [[0], [1]]
rede = Rede(0.01, data_vector, labels)
# para testar a inicialização aleatória, basta mudar o valor do atributo weights_initialization_mode para "random". O valor "zeros" é o padrão.
rede.weights_initialization_mode = "random"
rede.create_initial_layer(4, "sigmoid")
rede.create_hidden_layer(8, "relu")
rede.create_hidden_layer(1, "sigmoid")


print("testando feedforward")
# define a função de custo como mean_squared_error
rede.set_cost_function("mean_squared_error")
# exemplo de vetor de entrada para a rede
input_vector = np.array([0.5, 0.2, 0.1, 0.2])
y_true_vector = np.array([1])  # exemplo de vetor de saída verdadeira
output_vector, loss = rede.feedforward(input_vector, y_true_vector)
print("output_vector: ", output_vector)
print("loss: ", loss)
