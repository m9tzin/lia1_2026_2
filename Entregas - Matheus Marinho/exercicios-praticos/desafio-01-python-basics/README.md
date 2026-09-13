# 🎬 Desafio Final: Análise Exploratória do IMDB Top 1000

**Aluno:** Matheus Sousa Marinho · **Matrícula:** 202206132
**Disciplina:** LIA 1 (2026/2)

## A pergunta

> Entre filmes que já são aclamados pela crítica e pelo público, o que separa os
> campeões de bilheteria dos demais?

O dataset reúne os 1000 títulos mais bem avaliados do IMDb, ou seja, uma amostra já filtrada por
qualidade. Por isso a pergunta não é "o que faz um filme faturar", que estes dados não
conseguem responder, mas o que diferencia os campeões **dentro** desse grupo de eleitos.

## A resposta

**Não é qualidade, e sim alcance.**

| Métrica | Demais | Campeões | Razão |
|---|---|---|---|
| Bilheteria mediana | US$ 9M | US$ 171M | **18,2×** |
| Votos medianos | 125.822 | 496.825 | **3,9×** |
| Nota IMDB mediana | 7,9 | 8,0 | 1,01× |
| Meta_score mediano | 78 | 78 | **1,00×** |

A correlação entre nota IMDB e bilheteria, em escala log, é de **−0,004** (zero; na escala
bruta, **+0,10**, que também é quase nada); entre votos e bilheteria, **+0,649**. O que
produz alcance é a categoria do filme: Aventura fatura 7,4× mais que Drama, e filmes de
classificação livre viram campeões em 36% dos casos contra 17% dos adultos.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `analise_imdb_top1000.ipynb` | Notebook completo, já executado (75 células, 11 gráficos) |
| `imdb_top_1000.csv` | Dataset do Kaggle |

## Estrutura do notebook

1. **Preparação**: carga dos dados (funciona local e no Colab)
2. **🧹 Limpeza**: conversão de tipos, valor corrompido, 16 categorias inconsistentes, campo multivalorado
3. **📊 Análise exploratória**: distribuições, correlações, panorama histórico
4. **📈 Visualizações**: os gráficos que sustentam o argumento
5. **🔍 Insights**: definição de "campeão" normalizada por década e perfil comparativo
6. **📋 Conclusões**: resposta, recomendações e limitações

## Como executar

**Google Colab**: abra o notebook e execute; ele pede o upload do CSV automaticamente.

**Local:**
```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook analise_imdb_top1000.ipynb
```

## Decisões metodológicas

- **"Campeão" é definido dentro de cada década** (top 25% da própria década). `Gross` é
  nominal, sem correção de inflação, e comparar 1970 com 2015 em dólares brutos seria erro.
- **Mediana em vez de média:** a bilheteria é fortemente assimétrica, e a média é puxada
  pelos poucos gigantes.
- **Escala logarítmica** nos gráficos de bilheteria, pela mesma razão.
- **Nenhuma linha foi descartada.** `Drishyam` aparece duas vezes, mas são dois filmes
  distintos (original de 2013 e remake de 2015), e um `drop_duplicates()` no título teria
  apagado um filme legítimo.

## Limitações declaradas

1. **Viés de seleção**: as conclusões valem entre filmes já aclamados, não para o cinema em geral.
2. **`Gross` é doméstico** (EUA/Canadá), verificado no notebook: `Titanic` aparece com US$ 659M contra ~US$ 2,2 bi mundiais.
3. **Bruto, não lucro**: sem orçamento no dataset, não há como calcular ROI.
4. **Correlação não é causalidade**: votos são um *proxy* de alcance, não sua causa.
5. **169 filmes (16,9%) sem bilheteria** ficaram fora das análises de faturamento.

## Fonte

[IMDB Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows), no Kaggle
