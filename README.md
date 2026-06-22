# Rede Neural do Zero — Biblioteca de MLP

Biblioteca de **redes neurais multicamadas (MLP) implementada do zero** em Python, sem frameworks de aprendizado de máquina (TensorFlow, PyTorch, scikit-learn). A única dependência usada para a álgebra linear é o `numpy`. Projeto final da disciplina de **Aprendizado de Máquina**.

> **A rede neural em si foi implementada usando apenas `numpy`.** As bibliotecas `pandas` e `matplotlib` são usadas **somente nos notebooks**, para manipulação dos conjuntos de dados e plotagem de gráficos — elas não fazem parte do desenvolvimento da rede neural.

Biblioteca desenvolvida por:
- Lucas Ladwig (22100910) - [@lucas.ladwig](https://codigos.ufsc.br/lucas.ladwig)
- Luis Felipe de Azambuja Feyh (22100913) - [@luis.felipe.feyh](https://codigos.ufsc.br/luis.felipe.feyh)
- Luiz Fernando Aguilar Althoff (21202336) - [@luiz.fernando.althoff](https://codigos.ufsc.br/luiz.fernando.althoff)

## Recursos

- **Arquitetura flexível** — número arbitrário de camadas, de neurônios por camada, função de ativação por camada e função de custo por rede.
- **8 funções de ativação** (além da identidade), cada uma com sua derivada para o backpropagation: `sigmoid`, `tanh`, `relu`, `leaky_relu`, `parametric_relu` (com inclinação `a` aprendível por neurônio), `elu`, `swish` e `softmax`.
- **3 funções de custo** — erro quadrático médio, entropia cruzada binária e entropia cruzada categórica.
- **2 inicializações de pesos** — `zeros` e `random`.
- **Algoritmos de treino** — `feedforward`, `back_propagation` e `gradient_descent`, orquestrados por um laço de treino (`train`).

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # numpy (rede neural); pandas e matplotlib (apenas nos notebooks)
```

## Uso rápido

O exemplo abaixo cria, treina e usa uma rede para aprender a função lógica **AND**:

```python
import numpy as np
from api.main import Rede

# 1. Dados como arrays numpy: X = (n_amostras, n_atributos), y = (n_amostras, n_saidas)
X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [0.0], [0.0], [1.0]])          # funcao logica AND

# 2. Cria a rede com a taxa de aprendizado e os dados de treino
rede = Rede(0.5, atributes=X, labels=y)
rede.weights_initialization_mode = "random"         # 'zeros' nao aprende (simetria entre neuronios)

# 3. Monta a arquitetura, uma camada por vez
rede.create_initial_layer(4, "relu")                # 1a camada oculta (tamanho da entrada vem de X)
rede.create_hidden_layer(1, "sigmoid")              # camada de saida

# 4. Define a funcao de custo
rede.set_cost_function("mean_squared_error")

# 5. Treina; retorna o historico de perda media por epoca
historico = rede.train(epochs=500)
print(f"perda: {historico[0]:.4f} -> {historico[-1]:.4f}")

# 6. Previsao em uma nova entrada (acrescente o slot do bias = 1.0)
entrada = np.append(X[3], 1.0)
previsao, _ = rede.feedforward(entrada, np.zeros(1))
print("previsao:", previsao)                        # proximo de 1.0
```

## Arquitetura da API

A biblioteca inteira está na classe **`Rede`** (`api/main.py`). A rede é montada de forma **incremental**, adicionando uma camada de cada vez:

| Etapa | Método | O que faz |
|---|---|---|
| Criar a rede | `Rede(learning_rate, atributes, labels)` | guarda a taxa de aprendizado e os dados de treino |
| Primeira camada | `create_initial_layer(n, ativacao)` | descobre o tamanho da entrada a partir dos atributos |
| Demais camadas | `create_hidden_layer(n, ativacao)` | adiciona camadas ocultas e a de saída |
| Custo | `set_cost_function(nome)` | define a função de perda da rede |
| Treino | `train(epocas)` | laço `feedforward → back_propagation → gradient_descent`; devolve o histórico de perda |

Dois pontos de projeto importantes:

- **Padrão de registro:** a `Rede` mantém dicionários que mapeiam o nome (string) de cada ativação, derivada e custo para o método correspondente, permitindo escolher os componentes por nome.
- **"Bias trick":** o viés é embutido como a última coluna da matriz de pesos de cada camada, pareado com uma entrada constante `1.0` — por isso a inferência manual acrescenta `1.0` ao vetor de entrada.

### Funções de ativação

| Ativação | Nome (string) |
|---|---|
| Sigmoid | `"sigmoid"` |
| Tangente hiperbólica | `"tanh"` |
| ReLU | `"relu"` |
| Leaky ReLU | `"leaky_relu"` |
| Parametric ReLU | `"parametric_relu"` |
| ELU | `"elu"` |
| Swish | `"swish"` |
| Softmax | `"softmax"` |

> A ativação **identidade/linear** (`"linear"`) existe, mas por ser uma função pronta não conta para o requisito de 8 ativações.

### Funções de custo

| Custo | Nome (string) | Uso típico |
|---|---|---|
| Erro quadrático médio | `"mean_squared_error"` | regressão / binária |
| Entropia cruzada binária | `"binary_cross_entropy"` | classificação binária (saída sigmoide) |
| Entropia cruzada categórica | `"categoric_cross_entropy"` | classificação multiclasse (saída softmax) |

### Inicialização de pesos

```python
rede.weights_initialization_mode = "random"   # ou "zeros"
```

## Estrutura do repositório

- `api/` — a biblioteca (classe `Rede`) e componentes auxiliares.
- `datasets/` — conjuntos de dados: brutos em `_raw/`, limpos e normalizados em `clean/`.
- `notebooks/` — demonstrações de ponta a ponta das três tarefas.
- `images/` — diagramas de projeto usados neste README.
- `arquivo_de_testes.py` — testes da API (valores de referência e *gradient checking*).

Para rodar os testes (a partir da raiz do repositório):

```bash
python arquivo_de_testes.py
```

## Notebooks

Três tarefas demonstradas de ponta a ponta (carregamento → divisão treino/validação/teste → treino → ajuste na validação → avaliação no teste, com texto explicativo, métricas e gráficos):

- `notebooks/binary.ipynb` — **classificação binária** (câncer de mama, Wisconsin).
- `notebooks/muliclass.ipynb` — **classificação multiclasse** (Iris).
- `notebooks/regression.ipynb` — **regressão** (preços de imóveis).

## Limitações

- O treino usa **descida de gradiente estocástica** (uma amostra por vez), sem *mini-batches* nem embaralhamento entre épocas.
- A **softmax** deve ser usada com **entropia cruzada categórica** (a derivada combinada é simplificada no backpropagation).
- O **erro quadrático médio** considera saída de dimensão 1 (regressão/binária).
- As entropias cruzadas não fazem *clipping* numérico: saídas saturadas (exatamente 0 ou 1) podem gerar perda `NaN`, embora o gradiente (`ŷ − y`) e o próprio treino não sejam afetados.
- A inicialização de pesos cobre apenas `zeros` e `random` (sem Xavier/He).
- Não há método `predict` dedicado: a inferência usa `feedforward` (acrescentando o slot do bias `1.0`).
