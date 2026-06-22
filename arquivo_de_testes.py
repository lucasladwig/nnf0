# UTILIZE ESTE ARQUIVO PARA TESTES UNITÁRIOS DA API!

from api.main import Rede
import numpy as np


def test_backpropagation_valores_de_referencia():
    """Compara os gradientes da back_propagation com valores conhecidos.

    Rede: 1 atributo -> 2 neurônios sigmoide (oculta) -> 1 neurônio sigmoide (saída),
    pesos fixos, alvo y=1.0, custo erro quadrático. Os gradientes esperados foram
    calculados em precisão total e conferidos por diferenças finitas.
    """
    rede = Rede(0.1, atributes=np.zeros((1, 1)), labels=np.zeros((1, 1)))
    rede.create_initial_layer(2, "sigmoid")  # camada oculta -> matriz (2, 2)
    rede.create_hidden_layer(1, "sigmoid")   # camada de saída -> matriz (1, 3)

    # sobrescreve os pesos com os valores exatos do exemplo de referência
    rede.network[0] = np.array([[0.10, 0.00], [0.20, 0.10]])
    rede.network[1] = np.array([[0.30, 0.40, 0.10]])
    rede.set_cost_function("mean_squared_error")

    x = np.array([0.5, 1.0])  # 1 atributo + slot do bias (=1.0)
    y = np.array([1.0])

    prediction, _ = rede.feedforward(x, y)
    gradients = rede.back_propagation(prediction, y)

    esperado_saida = np.array([[-0.09301810, -0.09979468, -0.18149966]])
    esperado_oculta = np.array([[-0.00680199, -0.01360397],
                                [-0.00898483, -0.01796967]])

    assert np.allclose(gradients[1], esperado_saida, atol=1e-6), gradients[1]
    assert np.allclose(gradients[0], esperado_oculta, atol=1e-6), gradients[0]
    print("[OK] gradientes batem com os valores de referência")


def test_backpropagation_gradient_check():
    """Gradient checking: confere os gradientes analíticos da back_propagation
    contra diferenças finitas numéricas, em uma rede com pesos aleatórios.
    Não depende de valores fixos — vale para qualquer arquitetura.
    """
    np.random.seed(0)
    rede = Rede(0.1, atributes=np.zeros((1, 3)), labels=np.zeros((1, 1)))
    rede.weights_initialization_mode = "random"
    rede.create_initial_layer(4, "sigmoid")
    rede.create_hidden_layer(2, "sigmoid")
    rede.create_hidden_layer(1, "sigmoid")
    rede.set_cost_function("mean_squared_error")

    x = np.array([0.5, -0.2, 0.1, 1.0])  # 3 atributos + slot do bias
    y = np.array([1.0])

    prediction, _ = rede.feedforward(x, y)
    gradients = rede.back_propagation(prediction, y)

    epsilon = 1e-5
    maior_diferenca = 0.0
    for layer_index in range(len(rede.network)):
        layer = rede.network[layer_index]
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                peso_original = layer[i, j]
                layer[i, j] = peso_original + epsilon
                loss_mais = rede.feedforward(x, y)[1]
                layer[i, j] = peso_original - epsilon
                loss_menos = rede.feedforward(x, y)[1]
                layer[i, j] = peso_original  # restaura o peso
                gradiente_numerico = (loss_mais - loss_menos) / (2 * epsilon)
                gradiente_analitico = gradients[layer_index][i, j]
                maior_diferenca = max(maior_diferenca,
                                      abs(gradiente_numerico - gradiente_analitico))
    print(f"[OK] gradient check: maior diferença analítico vs numérico = {maior_diferenca:.2e}")
    assert maior_diferenca < 1e-6, maior_diferenca


def test_softmax_camada_de_saida():
    """A softmax deve operar sobre o vetor de pré-ativações da camada inteira,
    produzindo uma distribuição de probabilidade (soma = 1), e não neurônio a
    neurônio. Antes da correção, cada neurônio recebia um escalar e softmax(escalar)
    = 1.0, fazendo a saída somar o número de neurônios em vez de 1.
    """
    rede = Rede(0.1, atributes=np.zeros((1, 2)), labels=np.zeros((1, 3)))
    rede.create_initial_layer(3, "softmax")  # camada única de saída -> matriz (3, 3)
    rede.network[0] = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])
    rede.set_cost_function("categoric_cross_entropy")

    x = np.array([1.0, 2.0, 1.0])  # 2 atributos + slot do bias (=1.0)
    y = np.array([0.0, 1.0, 0.0])

    prediction, _ = rede.feedforward(x, y)

    z = rede.network[0].dot(x)
    esperado = np.exp(z) / np.sum(np.exp(z))

    assert np.isclose(np.sum(prediction), 1.0), np.sum(prediction)
    assert np.allclose(prediction, esperado, atol=1e-9), prediction
    print("[OK] softmax produz uma distribuição de probabilidade na camada de saída")


def test_parametric_relu_forward():
    """No forward, a parametric ReLU passa z>0 direto e multiplica z<0 pelo alpha
    (a) daquele neurônio. Cada neurônio tem seu próprio alpha aprendível.
    """
    rede = Rede(0.1, atributes=np.zeros((1, 2)), labels=np.zeros((1, 2)))
    rede.create_initial_layer(2, "parametric_relu")  # camada única -> matriz (2, 3)
    rede.network[0] = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]])
    rede.param_relu_alphas[0] = np.array([0.1, 0.25])
    rede.set_cost_function("mean_squared_error")

    x = np.array([2.0, -4.0, 1.0])  # pré-ativações resultantes: z = [2.0, -4.0]
    y = np.array([0.0, 0.0])

    prediction, _ = rede.feedforward(x, y)

    # z0 = 2.0 > 0 -> 2.0 ; z1 = -4.0 < 0 -> alpha=0.25 * -4.0 = -1.0
    esperado = np.array([2.0, -1.0])
    assert np.allclose(prediction, esperado), prediction
    print("[OK] parametric_relu aplica o alpha de cada neurônio nas entradas negativas")


def test_parametric_relu_gradient_check():
    """Gradient checking da parametric ReLU: confere por diferenças finitas tanto
    os gradientes dos pesos quanto os gradientes do parâmetro aprendível 'a'
    (alpha) de cada neurônio. Garante que a PReLU está integrada ao forward e à
    backpropagation, com o gradiente do alpha pronto para o gradient_descent.
    """
    rede = Rede(0.1, atributes=np.zeros((1, 2)), labels=np.zeros((1, 1)))
    rede.create_initial_layer(3, "parametric_relu")  # camada oculta PReLU -> (3, 3)
    rede.create_hidden_layer(1, "sigmoid")           # camada de saída sigmoide -> (1, 4)
    rede.set_cost_function("mean_squared_error")

    # pesos e alphas fixos, escolhidos para gerar pré-ativações de sinais variados
    rede.network[0] = np.array([[0.5, -0.3, 0.1],
                                [-0.4, 0.2, -0.6],
                                [0.3, 0.7, -0.2]])
    rede.network[1] = np.array([[0.2, -0.5, 0.4, 0.1]])
    rede.param_relu_alphas[0] = np.array([0.1, 0.25, 0.5])

    x = np.array([0.7, -0.4, 1.0])  # 2 atributos + slot do bias
    y = np.array([1.0])

    prediction, _ = rede.feedforward(x, y)
    gradients = rede.back_propagation(prediction, y)

    epsilon = 1e-5
    maior_diferenca = 0.0

    # 1) gradientes dos pesos
    for layer_index in range(len(rede.network)):
        layer = rede.network[layer_index]
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                original = layer[i, j]
                layer[i, j] = original + epsilon
                loss_mais = rede.feedforward(x, y)[1]
                layer[i, j] = original - epsilon
                loss_menos = rede.feedforward(x, y)[1]
                layer[i, j] = original
                gradiente_numerico = (loss_mais - loss_menos) / (2 * epsilon)
                maior_diferenca = max(maior_diferenca,
                                      abs(gradiente_numerico - gradients[layer_index][i, j]))

    # 2) gradientes do parâmetro aprendível 'a' (alpha) da camada PReLU
    alphas = rede.param_relu_alphas[0]
    for k in range(alphas.shape[0]):
        original = alphas[k]
        alphas[k] = original + epsilon
        loss_mais = rede.feedforward(x, y)[1]
        alphas[k] = original - epsilon
        loss_menos = rede.feedforward(x, y)[1]
        alphas[k] = original
        gradiente_numerico = (loss_mais - loss_menos) / (2 * epsilon)
        maior_diferenca = max(maior_diferenca,
                              abs(gradiente_numerico - rede.param_relu_alpha_gradients[0][k]))

    print(f"[OK] PReLU gradient check (pesos + alpha): maior diferença = {maior_diferenca:.2e}")
    assert maior_diferenca < 1e-6, maior_diferenca


def test_gradient_descent_reduz_loss():
    """Um passo de gradient_descent deve mover os parâmetros na direção oposta ao
    gradiente, reduzindo a perda na mesma amostra. Verifica que atualiza tanto os
    pesos quanto os alphas aprendíveis da parametric_relu (e ignora as camadas sem
    alpha, cujo gradiente fica None).
    """
    rede = Rede(0.1, atributes=np.zeros((1, 2)), labels=np.zeros((1, 1)))
    rede.create_initial_layer(3, "parametric_relu")  # camada oculta PReLU -> (3, 3)
    rede.create_hidden_layer(1, "sigmoid")           # camada de saída sigmoide -> (1, 4)
    rede.set_cost_function("mean_squared_error")

    rede.network[0] = np.array([[0.5, -0.3, 0.1],
                                [-0.4, 0.2, -0.6],
                                [0.3, 0.7, -0.2]])
    rede.network[1] = np.array([[0.2, -0.5, 0.4, 0.1]])
    rede.param_relu_alphas[0] = np.array([0.1, 0.25, 0.5])

    x = np.array([0.7, -0.4, 1.0])  # 2 atributos + slot do bias
    y = np.array([1.0])

    prediction, loss_antes = rede.feedforward(x, y)
    gradients = rede.back_propagation(prediction, y)

    pesos_antes = rede.network[0].copy()
    alphas_antes = rede.param_relu_alphas[0].copy()

    rede.gradient_descent(gradients)

    # os pesos e os alphas (das entradas negativas) foram de fato atualizados
    assert not np.allclose(rede.network[0], pesos_antes), "pesos não foram atualizados"
    assert not np.allclose(rede.param_relu_alphas[0], alphas_antes), "alphas não foram atualizados"

    # a perda na mesma amostra diminuiu após o passo
    _, loss_depois = rede.feedforward(x, y)
    assert loss_depois < loss_antes, (loss_antes, loss_depois)
    print(f"[OK] gradient_descent reduziu a perda: {loss_antes:.6f} -> {loss_depois:.6f}")


if __name__ == "__main__":
    test_backpropagation_valores_de_referencia()
    test_backpropagation_gradient_check()
    test_softmax_camada_de_saida()
    test_parametric_relu_forward()
    test_parametric_relu_gradient_check()
    test_gradient_descent_reduz_loss()
    print("Todos os testes passaram.")
