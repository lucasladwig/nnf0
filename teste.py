from api.main import Rede
import numpy as np


data = [[0.5, 0.2, 0.1], [0.9, 0.8, 0.7]] # imagine que aqui tem um dataframe, porém sem a coluna de resposta
data_vector = np.array(data) # converti para np.array porque no código ele pega o tamanho por shape()
labels = [[0], [1]] # imagine que aqui tem um dataframe, porém apenas com a coluna de resposta
rede = Rede(0.01, data_vector, labels)
rede.weights_initialization_mode = "random" # para testar a inicialização aleatória, basta mudar o valor do atributo weights_initialization_mode para "random". O valor "zeros" é o padrão.
print("testando criação de camadas")
rede.create_initial_layer(4, "sigmoid")
print(rede.network[0]) # todos os pesos da primeira camada (4 neurônios, cada um com 3 pesos + peso do bias)
print(rede.network[0][0]) # pesos do primeiro neurônio da primeira camada
print(rede.layers_activation_func_list)