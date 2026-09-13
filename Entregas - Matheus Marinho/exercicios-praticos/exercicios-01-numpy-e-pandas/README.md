# 📦 Exercícios Práticos: NumPy e Pandas

**Aluno:** Matheus Sousa Marinho · **Matrícula:** 202206132
**Disciplina:** LIA 1 (2026/2)

Resolução dos três desafios propostos ao final de
`1_1_Python_Basics_A1_Numpy_e_Pandas.ipynb`.

## O que mudou as respostas

Os dados do Desafio 1 são gerados pelo próprio notebook da aula, e a forma como são
construídos limita o que as perguntas conseguem medir. Duas coisas apareceram antes de
qualquer cálculo:

**1. `Produto` e `Regiao` são a mesma variável com dois nomes.** As duas listas têm
ciclo de 5 e são repetidas 6 vezes, então a posição `i` sempre casa o mesmo produto com
a mesma região. Só 5 das 25 combinações possíveis existem:

| Produto | Região |
|---|---|
| Notebook | Norte |
| Mouse | Sul |
| Teclado | Nordeste |
| Monitor | Centro-Oeste |
| Headset | Sudeste |

Na prática, `groupby("Regiao")` e `groupby("Produto")` devolvem os mesmos números com
rótulos trocados. As perguntas 2 e 3 do enunciado interrogam o mesmo eixo.

**2. Os vendedores não recebem o mesmo número de registros.** A lista repete `Ana`
quatro vezes a cada dez posições, então ela entra com 12 registros contra 9 de João e
9 de Pedro, um terço a mais de oportunidades antes de qualquer venda ser sorteada.

## Respostas do Desafio 1

| Pergunta | Resposta | Ressalva |
|---|---|---|
| 1. Vendedor com maior número de vendas | **Ana** em volume: 12 vendas, R$ 66.264 e 156 unidades. **Pedro** em ticket médio: R$ 5.677,56 | O pódio se inverte conforme a métrica, e a vantagem de Ana vem das 3 vendas a mais |
| 2. Região com maior faturamento total | **Sudeste**, R$ 35.457 (22,4% do total) | Fica 0,12% à frente do Norte; a simulação mostra que é ruído |
| 3. Produto mais vendido em quantidade | **Notebook**, 70 unidades | Não é o que mais fatura, porque valor e quantidade são independentes nestes dados |
| 4. Média de vendas por vendedor | Pedro R$ 5.677,56, Ana R$ 5.522,00, João R$ 4.525,67 | Média geral de R$ 5.269,77 |

Sobre a pergunta 2: a vantagem do Sudeste sobre o Norte é de R$ 43 em R$ 35 mil. Para
medir quanto disso é sorteio, o notebook refaz o experimento 10.000 vezes mantendo a
estrutura dos dados e sorteando apenas os valores. Cada região vence perto de 20% das
vezes, que é o esperado se elas forem indistinguíveis. **Não há região líder**: o
conjunto foi construído sem nenhuma diferença real entre elas.

## Desafio 2: limpeza

Um cadastro de clientes com 9 tipos de defeito plantados de propósito, de 12 para 9
registros ao final.

| Defeito | Tratamento |
|---|---|
| Nulos disfarçados (`""`, `"N/A"`, `"?"`) | Convertidos em `NaN` antes de tudo, para que as contagens parem de mentir |
| Espaços sobrando e internos repetidos | `str.strip()` e `str.replace(r"\s+", " ")` |
| Caixa inconsistente | `title()` para nomes e cidades, `upper()` para UF, `lower()` para e-mail |
| Linha duplicada de ponta a ponta | `drop_duplicates()` |
| Duplicata que só aparece após padronizar | Resolvida pela ordem das etapas, com contrafactual que prova a decisão (seção 3.3) |
| Moeda como texto (`"R$ 7.100,00"`) | Separadores trocados na ordem certa, depois `to_numeric` |
| Datas em dois formatos no mesmo campo | Conversão explícita, formato a formato |
| Idade ausente | Preenchida com a mediana |
| Cidade e UF ausentes | Marcadas como `"Não informado"`, sem inventar localidade |

### Duas decisões que mudaram o resultado

**Padronizar antes de deduplicar.** O enunciado lista a remoção de duplicatas primeiro,
mas enquanto `"carlos EDUARDO lima"` e `"Carlos Eduardo Lima "` forem strings
diferentes, `drop_duplicates()` não vê duplicata nenhuma. O notebook prova isso com um
contrafactual na seção 3.3: na ordem do enunciado, o cliente 2 sobrevive em duplicata e passa a ser
contado duas vezes em toda soma.

**Converter as datas formato a formato.** A saída óbvia,
`pd.to_datetime(..., format="mixed", dayfirst=True)`, aplica `dayfirst` também às datas
ISO, e o que o pandas faz com essa contradição mudou entre versões: no 3.0.5,
`"2024-03-02"` vira 3 de fevereiro sem levantar erro; no 2.2.3, sai correto. O notebook
executa os dois caminhos e compara na versão que estiver rodando.

## Desafio 3: visualização

| Gráfico | O que mostra |
|---|---|
| Barras em dois painéis | O pódio dos vendedores se inverte entre faturamento total e ticket médio |
| Mapa de calor por vendedor e região | As 5 combinações que não existem, e os totais por região na base |
| Histograma com diagrama de caixa | Distribuição uniforme e simétrica, sem valores extremos, que é a assinatura de `np.random.randint` |

O enunciado pede "tendências por região", mas o conjunto **não tem coluna de tempo** e
sem tempo não existe tendência a traçar. Em vez de fabricar um eixo temporal, o
notebook mostra a composição do faturamento por região, que é o que estes dados
sustentam.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `exercicios_numpy_pandas.ipynb` | Notebook completo, já executado (46 células, 3 gráficos) |
| `pyproject.toml` | Versões usadas, alinhadas com o Google Colab |

## Como executar

**Google Colab:** abra o notebook e execute. Todos os dados são gerados pelo próprio
código, então não há arquivo para enviar.

**Local, com [uv](https://docs.astral.sh/uv/):**

Esta entrega faz parte do workspace `uv` da pasta `Entregas - Matheus Marinho`,
que mantém um único `.venv` compartilhado por todas as entregas:

```bash
cd ..            # raiz do workspace
uv sync --all-packages
cd exercicios-01-numpy-e-pandas
```

Feito isso, `uv run` executa dentro do `.venv` do workspace, sem precisar ativar nada:

```bash
uv run jupyter notebook exercicios_numpy_pandas.ipynb
```

Para reexecutar o notebook inteiro sem abrir a interface:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace exercicios_numpy_pandas.ipynb
```

Se o `uv` não estiver instalado: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Reprodutibilidade

- `np.random.seed(42)` reproduz exatamente os dados de vendas da aula.
- Semente 0 na simulação de 10.000 sorteios.
- Data de referência fixa em `2025-01-01` nas colunas derivadas, em vez de
  `Timestamp.today()`, que faria os números mudarem a cada execução.

Executado de ponta a ponta sem erros nem avisos em **pandas 2.2.3** (versões deste
`pyproject.toml`, que são as do Colab) e também em **pandas 3.0.5** com numpy 2.5.2.

## Limitações declaradas

1. **Os dados de vendas são sintéticos.** Nada aqui descreve vendedores, produtos ou
   regiões de verdade.
2. **`Produto` e `Regiao` são redundantes**, então não há como separar o efeito de uma
   do efeito da outra.
3. **A amostra é minúscula**: 30 registros, 6 por região e de 9 a 12 por vendedor.
   Quase nenhuma diferença observada sobrevive a um segundo sorteio.
4. **O cadastro do Desafio 2 também é inventado**, construído para exibir defeitos
   específicos. Os defeitos são realistas; as pessoas e os valores, não.
5. **Os valores preenchidos não são medições.** A mediana das idades e o rótulo
   `"Não informado"` marcam onde falta dado e devem continuar sendo tratados assim.
