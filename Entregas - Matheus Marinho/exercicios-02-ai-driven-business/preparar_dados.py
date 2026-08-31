"""
Gera atletas.csv a partir dos dados brutos do Campeonato Brasileiro.

Os arquivos de origem sao registros de eventos (um gol por linha, um cartao por
linha), nao uma base de atletas. Este script agrega tudo em uma linha por atleta e
traduz as contagens em faixas qualitativas, no mesmo formato da base de leads da
aula, onde as colunas sao rotulos como "alto" e "baixo" em vez de numeros.

Os CSVs sao lidos direto do repositorio publico, sem copia local.

Fonte: https://github.com/adaoduque/Brasileirao_Dataset
Uso:   uv run python preparar_dados.py
"""

import pandas as pd

TEMPORADA = 2024
MIN_GOLS = 10         # os 12 goleadores de dois digitos da temporada
FONTE = "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master"
SAIDA = "atletas.csv"


def ler(nome):
    return pd.read_csv(f"{FONTE}/{nome}.csv")


gols = ler("campeonato-brasileiro-gols")
cartoes = ler("campeonato-brasileiro-cartoes")
partidas = ler("campeonato-brasileiro-full")

# 1. Recorte da temporada. O ano so existe nas partidas, e chega aos eventos pelo id.
partidas["data"] = pd.to_datetime(partidas["data"], format="%d/%m/%Y", errors="coerce")
partidas_ano = partidas[partidas["data"].dt.year == TEMPORADA]
ids_ano = set(partidas_ano["ID"])

gols_ano = gols[gols["partida_id"].isin(ids_ano)].copy()
cartoes_ano = cartoes[cartoes["partida_id"].isin(ids_ano)].copy()

# Gol contra nao e producao do atleta que marcou, entao sai da base.
gols_ano["tipo_de_gol"] = gols_ano["tipo_de_gol"].fillna("Normal")
gols_ano = gols_ano[gols_ano["tipo_de_gol"] != "Gol Contra"]

# 2. Uma linha por atleta. "quarto" divide as 38 rodadas em quatro blocos.
gols_ano["quarto"] = pd.cut(gols_ano["rodata"], bins=[0, 9, 19, 28, 38], labels=[1, 2, 3, 4])

base = (
    gols_ano.groupby("atleta")
    .agg(
        gols=("tipo_de_gol", "size"),
        penaltis=("tipo_de_gol", lambda s: (s == "Penalty").sum()),
        quartos_com_gol=("quarto", "nunique"),
        clube=("clube", lambda s: s.mode().iat[0]),
    )
    .reset_index()
)

base["amarelos"] = base["atleta"].map(
    cartoes_ano[cartoes_ano["cartao"] == "Amarelo"].groupby("atleta").size()
).fillna(0).astype(int)
base["vermelhos"] = base["atleta"].map(
    cartoes_ano[cartoes_ano["cartao"] == "Vermelho"].groupby("atleta").size()
).fillna(0).astype(int)
# Posicao so existe no arquivo de cartoes: quem nunca foi advertido fica sem ela.
base["posicao"] = base["atleta"].map(
    cartoes_ano.dropna(subset=["posicao"]).groupby("atleta")["posicao"].agg(lambda s: s.mode().iat[0])
).fillna("Nao informada")

# 3. Classificacao final, derivada dos placares. Vale como custo de negociacao.
casa = partidas_ano.rename(columns={"mandante": "clube", "mandante_Placar": "pro",
                                    "visitante_Placar": "contra"})[["clube", "pro", "contra"]]
fora = partidas_ano.rename(columns={"visitante": "clube", "visitante_Placar": "pro",
                                    "mandante_Placar": "contra"})[["clube", "pro", "contra"]]
jogos = pd.concat([casa, fora], ignore_index=True)
jogos["pontos"] = (jogos["pro"] > jogos["contra"]) * 3 + (jogos["pro"] == jogos["contra"]) * 1

tabela = (
    jogos.groupby("clube")
    .agg(pontos=("pontos", "sum"), gp=("pro", "sum"), gc=("contra", "sum"))
    .reset_index()
)
tabela["saldo"] = tabela["gp"] - tabela["gc"]
tabela = tabela.sort_values(["pontos", "saldo", "gp"], ascending=False).reset_index(drop=True)
tabela["posicao_final"] = tabela.index + 1

# Confere contra o resultado real de 2024 antes de seguir.
assert tabela.loc[0, "clube"] == "Botafogo-RJ" and tabela.loc[0, "pontos"] == 79
assert set(tabela.loc[tabela["posicao_final"] > 16, "clube"]) == {
    "Athletico-PR", "Criciuma", "Cuiaba", "Atletico-GO"
}

# 4. Shortlist e traducao em faixas.
# Os cortes sao fixos e explicitos, nao quantis: com uma dezena de atletas os tercis
# caem todos no mesmo valor e as faixas deixam de separar qualquer coisa.
df = base[base["gols"] >= MIN_GOLS].copy().reset_index(drop=True)
df = df.merge(tabela[["clube", "posicao_final"]], on="clube", how="left")

# Todos os selecionados sao goleadores de dois digitos, entao nenhuma faixa se chama
# "baixo": chamar 10 gols de baixo volume daria ao modelo uma informacao falsa.
df["volume_ofensivo"] = pd.cut(df["gols"], [-1, 10, 12, 1e9],
                               labels=["bom", "alto", "muito alto"]).astype(str)

df["regularidade"] = df["quartos_com_gol"].map(
    {4: "constante", 3: "regular", 2: "concentrada", 1: "concentrada"}
)

df["dependencia_de_penalti"] = pd.cut(df["penaltis"] / df["gols"], [-0.01, 0.0, 0.25, 1.0],
                                      labels=["nula", "baixa", "alta"]).astype(str)

pontos_disciplina = df["amarelos"] + 3 * df["vermelhos"]   # vermelho vale o triplo
df["risco_disciplinar"] = pd.cut(pontos_disciplina, [-1, 4, 8, 1e9],
                                 labels=["baixo", "medio", "alto"]).astype(str)

df["dificuldade_negociacao"] = pd.cut(df["posicao_final"], [0, 6, 16, 20],
                                      labels=["alta", "media", "baixa"]).astype(str)

df.insert(0, "id_atleta", range(1, len(df) + 1))
colunas = ["id_atleta", "atleta", "clube", "posicao", "volume_ofensivo", "regularidade",
           "dependencia_de_penalti", "risco_disciplinar", "dificuldade_negociacao"]

assert not df[colunas].isin(["nan"]).any().any(), "alguma faixa ficou sem rotulo"

df[colunas].to_csv(SAIDA, index=False, encoding="utf-8")
print(f"{SAIDA}: {len(df)} atletas com {MIN_GOLS}+ gols na Serie A {TEMPORADA}\n")
for coluna in colunas[4:]:
    print(f"{coluna:24s} {df[coluna].value_counts().to_dict()}")
