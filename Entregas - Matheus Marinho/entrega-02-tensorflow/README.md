# ♻️ Estimativa Direta da Composição de Lotes Recicláveis

**Aluno:** Matheus Sousa Marinho · **Matrícula:** 202206132 · **Disciplina:** LIA 1 (2026/2)

Este projeto usa classificação de imagens para estimar a composição de um lote de materiais
recicláveis. Cada fotografia é classificada como papelão, vidro, metal, papel, plástico ou rejeito.
Em seguida, as previsões são contadas e convertidas em porcentagens do lote.

Se 35 de 100 imagens forem classificadas como papel, a composição estimada será 35% de papel. Como
já conhecemos a composição verdadeira dos lotes simulados, podemos compará-la diretamente com a
estimativa.

## 🎯 O que o projeto faz

- baixa o dataset público TrashNet de um espelho no Hugging Face, sem autenticação;
- separa os dados em treino, validação e teste de forma estratificada;
- usa a EfficientNetB0 pré-treinada no ImageNet como extrator de características congelado;
- treina uma camada final `Dense` com softmax para as seis classes;
- avalia o classificador com acurácia, precisão, recall, F1-score e matriz de confusão;
- simula seis tipos de lote e estima a composição pela contagem direta das previsões;
- mede o erro médio por material em pontos percentuais;
- gera tabelas, gráficos e arquivos CSV com os resultados.

O notebook é autocontido: não depende de módulos Python auxiliares e pode ser executado diretamente
no Google Colab.

## 🗂️ Dataset

O [TrashNet](https://github.com/garythung/trashnet) possui 2.527 imagens distribuídas em seis
classes:

| material | imagens | proporção |
| --- | ---: | ---: |
| papelão | 403 | 15,9% |
| vidro | 501 | 19,8% |
| metal | 410 | 16,2% |
| papel | 594 | 23,5% |
| plástico | 482 | 19,1% |
| rejeito | 137 | 5,4% |

O notebook baixa o arquivo por `tf.keras.utils.get_file` a partir do espelho público no Hugging
Face. O conjunto fica em cache na pasta `dados/`.

## 📁 Estrutura

| arquivo | finalidade |
| --- | --- |
| `triagem_composicao_lote.ipynb` | implementação completa, com código e resultados visíveis |
| `test_notebook.py` | cinco testes das funções de composição e da sintaxe das células |
| `saidas/bancada_cenarios.csv` | erro médio e desvio-padrão em cada cenário |
| `saidas/relatorio_lote.csv` | composição verdadeira e estimada no lote de exemplo |
| `saidas/matriz_confusao.png` | matriz de confusão do classificador |
| `saidas/erro_por_cenario.png` | erro da estimativa de composição por cenário |
| `saidas/lote_exemplar.png` | comparação entre composição verdadeira e estimada |

## ▶️ Como executar

### 💻 Localmente

Na raiz do workspace das entregas:

```bash
uv sync --all-packages
cd "entrega-02-tensorflow"
uv run pytest -q
uv run jupyter notebook triagem_composicao_lote.ipynb
```

Na primeira execução são baixados o dataset e os pesos da EfficientNetB0. Com ambos em cache, a
execução completa levou menos de um minuto em CPU na máquina usada para esta entrega.

### ☁️ Google Colab

1. Envie ou abra `triagem_composicao_lote.ipynb` no Colab.
2. Selecione **Ambiente de execução → Executar tudo**.
3. Mantenha o acesso à internet habilitado para baixar o dataset e os pesos da rede.

Nenhum arquivo auxiliar precisa ser enviado. Se alguma operação determinística não estiver
disponível na GPU, altere `DETERMINISMO = True` para `False` na célula de configuração. Essa mudança
pode causar pequenas variações entre execuções, mas não altera o método.

## 🧮 Como a composição é medida

Para um lote com `N` imagens:

1. o modelo prevê uma classe para cada imagem;
2. conta-se quantas previsões pertencem a cada classe;
3. cada contagem é dividida por `N`;
4. na avaliação, a porcentagem estimada é comparada com a porcentagem verdadeira.

O erro de um material é a diferença absoluta entre essas duas porcentagens. O número apresentado no
gráfico é a média dos erros dos seis materiais. Por exemplo, erro médio de 3 pontos significa que,
em média, cada porcentagem estimada ficou 3 pontos percentuais distante da verdadeira.

## 📊 Resultados medidos

O classificador alcançou:

- **87,86%** de acurácia na validação;
- **90,00%** de acurácia nas 380 imagens de teste;
- F1-score macro de **0,890**.

Cada cenário foi repetido 30 vezes com lotes simulados de 200 imagens:

| cenário | erro médio | desvio-padrão |
| --- | ---: | ---: |
| equilibrado | 2,80 pontos | 0,53 ponto |
| dominado por papel | 1,62 ponto | 0,32 ponto |
| pouco vidro | 3,72 pontos | 0,86 ponto |
| sem metal | 3,77 pontos | 0,50 ponto |
| realista | 1,55 ponto | 0,47 ponto |
| muito rejeito | 5,78 pontos | 0,64 ponto |
| **média geral** | **3,21 pontos** | — |

No lote de exemplo, com 240 imagens e composição realista, o erro médio foi de **1,11 ponto
percentual por material**.

O cenário com muito rejeito foi o mais difícil. Essa classe possui apenas 137 imagens e apresentou
recall de 70% no teste, o que explica parte do erro maior.

## ⚠️ Limitações

- A estimativa herda diretamente os erros do classificador; não há correção matemática posterior.
- Cada fotografia do TrashNet contém um objeto centralizado sobre fundo simples, diferente de uma
  esteira real.
- Os lotes são simulados com reposição, então algumas imagens aparecem mais de uma vez.
- A classe rejeito tem poucas imagens em comparação às demais.
- Validação e teste pertencem ao mesmo dataset usado durante o desenvolvimento. Os números são uma
  avaliação interna, não uma garantia de desempenho em outro ambiente.
- Mudanças de câmera, iluminação, sujeira, deformação ou sobreposição de objetos podem reduzir a
  acurácia.

## 📚 Referências

- [TrashNet, dataset original](https://github.com/garythung/trashnet)
- [Espelho público no Hugging Face](https://huggingface.co/datasets/garythung/trashnet)
- [Keras: transfer learning](https://keras.io/guides/transfer_learning/)
- Tan e Le (2019), [EfficientNet](https://arxiv.org/abs/1905.11946)
