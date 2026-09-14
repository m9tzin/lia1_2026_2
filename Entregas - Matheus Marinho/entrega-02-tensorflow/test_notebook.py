"""Testes das funções simples usadas pelo notebook.

As funções continuam dentro do notebook para que ele seja autocontido no Colab.
Este arquivo executa a célula marcada como ``funcoes-lote`` e verifica seus
contratos com exemplos de resposta conhecida.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest


CAMINHO_NOTEBOOK = Path(__file__).with_name("triagem_composicao_lote.ipynb")
DIR_SAIDAS = Path(__file__).with_name("saidas")


def carregar_funcoes_lote() -> dict[str, object]:
    notebook = json.loads(CAMINHO_NOTEBOOK.read_text())
    celula = next(
        (
            celula
            for celula in notebook["cells"]
            if "funcoes-lote" in celula.get("metadata", {}).get("tags", [])
        ),
        None,
    )
    assert celula is not None, "célula funcoes-lote ausente do notebook"

    namespace: dict[str, object] = {"np": np}
    exec("".join(celula["source"]), namespace)
    return namespace


def test_todas_as_celulas_de_codigo_tem_sintaxe_valida():
    notebook = json.loads(CAMINHO_NOTEBOOK.read_text())

    for indice, celula in enumerate(notebook["cells"]):
        if celula["cell_type"] == "code":
            ast.parse("".join(celula["source"]), filename=f"célula {indice}")


def test_titulos_principais_usam_emojis():
    notebook = json.loads(CAMINHO_NOTEBOOK.read_text())
    texto_markdown = "".join(
        "".join(celula["source"])
        for celula in notebook["cells"]
        if celula["cell_type"] == "markdown"
    )

    titulos_esperados = (
        "# ♻️ Estimativa direta da composição de lotes recicláveis",
        "## 1. 🎯 Problema e proposta",
        "## 5. 🤖 Modelo de classificação",
        "## 6. 📊 Avaliação do classificador",
        "## 9. ✅ Conclusão e limitações",
    )

    for titulo in titulos_esperados:
        assert titulo in texto_markdown


@pytest.mark.parametrize("nome", ["bancada_cenarios.csv", "relatorio_lote.csv"])
def test_csvs_gerados_usam_quebra_de_linha_lf(nome):
    conteudo = (DIR_SAIDAS / nome).read_bytes()

    assert b"\r" not in conteudo


def test_composicao_conta_a_fracao_de_cada_classe():
    funcoes = carregar_funcoes_lote()

    resultado = funcoes["calcular_composicao"](np.array([0, 0, 1, 2]), 3)

    np.testing.assert_allclose(resultado, [0.5, 0.25, 0.25])


def test_composicao_recusa_lote_vazio():
    funcoes = carregar_funcoes_lote()

    with pytest.raises(ValueError, match="vazio"):
        funcoes["calcular_composicao"](np.array([], dtype=int), 3)


def test_erro_medio_e_exibido_em_pontos_percentuais():
    funcoes = carregar_funcoes_lote()

    erro = funcoes["erro_medio_percentual"](
        np.array([0.5, 0.3, 0.2]),
        np.array([0.4, 0.35, 0.25]),
    )

    assert erro == pytest.approx(100 * (0.10 + 0.05 + 0.05) / 3)


def test_montar_lote_respeita_tamanho_e_composicao():
    funcoes = carregar_funcoes_lote()
    rotulos = np.repeat([0, 1, 2], 10)

    indices = funcoes["montar_lote"](
        rotulos,
        np.array([0.5, 0.3, 0.2]),
        tamanho=10,
        gerador=np.random.default_rng(7),
    )

    assert len(indices) == 10
    np.testing.assert_allclose(
        funcoes["calcular_composicao"](rotulos[indices], 3),
        [0.5, 0.3, 0.2],
    )
