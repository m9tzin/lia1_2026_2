# ⚽ Agente Inteligente para Priorização de Alvos de Contratação

**Aluno:** Matheus Sousa Marinho · **Matrícula:** 202206132 · **Disciplina:** LIA 1 (2026/2)

O agente de priorização de leads da aula, aplicado a outro domínio: em vez de empresas
por potencial de compra de ERP, ele ranqueia atletas da Série A 2024 por prioridade de
contratação. Mesma estrutura de prompt, mesmos pesos explícitos, mesma saída em JSON,
com Gemini no lugar do Groq. Os 10 leads viram 12 artilheiros.

## Os dados

[Brasileirão Dataset](https://github.com/adaoduque/Brasileirao_Dataset), Série A 2024,
lido direto do repositório de origem. Os arquivos não são uma base de atletas: são
registros de eventos, uma linha por gol e uma linha por cartão. `preparar_dados.py`
cruza gols, cartões e partidas pela chave `partida_id` e monta uma linha por atleta,
com duas decisões:

- **Gol contra sai da conta**, senão o atleta é creditado por gols que fez contra o
  próprio time.
- **As contagens viram faixas antes do prompt.** A base de leads da aula chega ao modelo
  em rótulos (`alto`, `baixo`), não em números. Julgar rótulo é interpretar significado,
  que é onde um LLM ajuda; pedir média ponderada de números é usá-lo como calculadora,
  que é onde ele perde para uma linha de pandas.

O corte é de 10 gols ou mais, o que deixa 12 atletas de 324. A `dificuldade_negociacao`
sai da classificação final, derivada dos placares das 380 partidas, e o script confere o
resultado contra o fato conhecido antes de seguir.

## O ramo Gemini da aula não roda

O notebook original cria o modelo com o pacote antigo (`google.generativeai`) mas chama
`client.models.generate_content`, do pacote novo, e `client` nunca é definido: descomentar
aquele bloco levanta `NameError`. A correção aqui é pelo lado novo, com
`genai.Client(api_key=...)` do `google-genai`. O `gemini-2.5-flash` da aula também não
está mais disponível para chaves novas, então esta entrega usa `gemini-3.7-flash`.

## Os pesos

O prompt declara os cinco critérios e pede um score de 0 a 100 para cada atleta:

| Peso | Critério no prompt | Coluna |
|---|---|---|
| 35% | Volume ofensivo | `volume_ofensivo` |
| 20% | Regularidade ao longo da temporada | `regularidade` |
| 15% | Independência de pênalti | `dependencia_de_penalti` |
| 15% | Risco disciplinar | `risco_disciplinar` |
| 15% | Facilidade de negociação | `dificuldade_negociacao` |

Dois critérios são enunciados ao contrário da coluna, de propósito, para que os cinco
apontem no mesmo sentido e "mais" seja sempre melhor. Os percentuais são uma instrução ao
modelo, não uma fórmula executada: o score é estimado, não calculado.

## Resultado

Os 12 atletas vão em uma única chamada, como os 10 leads da aula, e os pesos de fato
interagiram em vez de o modelo só ordenar por gols:

| # | Atleta | Clube | Score | Por quê |
|---|---|---|---|---|
| 1 | Estêvão | Palmeiras | 86 | volume muito alto e regularidade constante, apesar da negociação cara |
| 2 | Pablo Vegetti | Vasco | 83 | nenhum gol de pênalti e clube mais acessível |
| 4 | Alerrandro | Vitória | 78 | artilheiro da temporada, puxado para baixo pela dependência de pênalti |

O artilheiro isolado do campeonato em 4º é o comportamento esperado de um critério
ponderado, não um erro: `dependencia_de_penalti` vale 15%. As limitações da base e do
método estão na última célula do notebook.

## Como executar

Chave do Gemini em https://aistudio.google.com/apikey.

```bash
cp .env.example .env       # e preencha GEMINI_API_KEY

cd ..                      # raiz do workspace uv
uv sync --all-packages
cd exercicios-02-ai-driven-business

uv run python preparar_dados.py                              # gera atletas.csv
uv run jupyter notebook agente_scouting_brasileirao.ipynb    # gera ranking_scouting_2024.csv
```

Esta entrega faz parte do workspace `uv` da pasta `Entregas - Matheus Marinho`, que
mantém um único `.venv` compartilhado por todas as entregas. O notebook precisa do
kernel desse `.venv` (Python 3.12), em `Entregas - Matheus Marinho/.venv/bin/python`.
Se o editor oferecer outro ambiente, o `import pandas` falha, e a correção é trocar o
kernel, não instalar pacotes de dentro do notebook.

Para regravar as saídas do notebook sem abrir a interface:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace agente_scouting_brasileirao.ipynb
```
