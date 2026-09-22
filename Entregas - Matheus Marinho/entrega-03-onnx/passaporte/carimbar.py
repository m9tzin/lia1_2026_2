"""Grava, lê e valida o passaporte de um modelo ONNX.

O passaporte é um JSON guardado em ``metadata_props`` na chave ``lia.passaporte``. Ele diz a
qualquer leitor como preparar a imagem, como interpretar a saída e qual decisão tomar, sem que o
leitor precise saber de qual framework o modelo veio.

Uso pela linha de comando:
    python -m passaporte.carimbar mostrar modelo.onnx
    python -m passaporte.carimbar gravar modelo.onnx passaporte.json [--destino saida.onnx]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema
import onnx

CHAVE = "lia.passaporte"
ESQUEMA = json.loads((Path(__file__).parent / "esquema_v1.json").read_text(encoding="utf-8"))


class PassaporteInvalido(ValueError):
    """O passaporte não segue o esquema ou contradiz o grafo do modelo."""


def _dims(valor_info: onnx.ValueInfoProto) -> list[int | None]:
    """Dimensões de um tensor do grafo; ``None`` para eixos dinâmicos."""
    return [
        d.dim_value if d.HasField("dim_value") else None
        for d in valor_info.type.tensor_type.shape.dim
    ]


def validar(passaporte: dict, modelo: onnx.ModelProto | None = None) -> None:
    """Valida o passaporte contra o esquema e, se houver modelo, contra o grafo."""
    try:
        jsonschema.validate(passaporte, ESQUEMA)
    except jsonschema.ValidationError as erro:
        caminho = "/".join(str(p) for p in erro.absolute_path) or "(raiz)"
        raise PassaporteInvalido(f"{caminho}: {erro.message}") from None

    classes = passaporte["classes"]
    if len(set(classes)) != len(classes):
        raise PassaporteInvalido("classes: há nomes repetidos")

    decisao = passaporte.get("decisao")
    if decisao:
        desconhecidas = set(decisao["alertar_se"]) - set(classes)
        if desconhecidas:
            raise PassaporteInvalido(
                f"decisao/alertar_se: classes fora da lista: {sorted(desconhecidas)}"
            )
        zona = decisao.get("zona_incerta")
        if zona and zona[0] > zona[1]:
            raise PassaporteInvalido("decisao/zona_incerta: início maior que o fim")

    formato = passaporte["saida"]["formato"]
    if formato in ("probabilidades", "logits") and passaporte["tarefa"] != "classificacao":
        raise PassaporteInvalido(f"saida/formato: {formato} exige tarefa classificacao")
    if formato.startswith("caixas") and passaporte["tarefa"] != "deteccao":
        raise PassaporteInvalido(f"saida/formato: {formato} exige tarefa deteccao")

    if modelo is not None:
        _validar_contra_grafo(passaporte, modelo)


def _validar_contra_grafo(passaporte: dict, modelo: onnx.ModelProto) -> None:
    entradas = {e.name: e for e in modelo.graph.input}
    saidas = {s.name: s for s in modelo.graph.output}
    nome_entrada = passaporte["entrada"]["tensor"]
    nome_saida = passaporte["saida"]["tensor"]

    if nome_entrada not in entradas:
        raise PassaporteInvalido(
            f"entrada/tensor: '{nome_entrada}' não existe; o grafo tem {sorted(entradas)}"
        )
    if nome_saida not in saidas:
        raise PassaporteInvalido(
            f"saida/tensor: '{nome_saida}' não existe; o grafo tem {sorted(saidas)}"
        )

    # Tamanho declarado precisa bater com o input quando ele for estático.
    dims_entrada = _dims(entradas[nome_entrada])
    if len(dims_entrada) == 4:
        altura, largura = passaporte["entrada"]["tamanho"]
        eixos = (2, 3) if passaporte["entrada"]["layout"] == "NCHW" else (1, 2)
        for eixo, esperado, rotulo in zip(eixos, (altura, largura), ("altura", "largura")):
            real = dims_entrada[eixo]
            if real is not None and real != esperado:
                raise PassaporteInvalido(
                    f"entrada/tamanho: {rotulo} declarada {esperado}, grafo tem {real}"
                )

    # Número de classes precisa bater com a saída quando ela for estática.
    nc = len(passaporte["classes"])
    dims_saida = _dims(saidas[nome_saida])
    formato = passaporte["saida"]["formato"]
    if formato == "caixas_xywh_por_classe" and len(dims_saida) == 3 and dims_saida[1]:
        if dims_saida[1] != 4 + nc:
            raise PassaporteInvalido(
                f"classes: {nc} declaradas, mas a saída tem {dims_saida[1]} canais "
                f"(esperado 4 + {nc} = {4 + nc})"
            )
    if formato in ("probabilidades", "logits") and dims_saida and dims_saida[-1]:
        if dims_saida[-1] != nc:
            raise PassaporteInvalido(
                f"classes: {nc} declaradas, mas a saída tem {dims_saida[-1]} valores"
            )


def ler(caminho_onnx: str | Path) -> dict | None:
    """Devolve o passaporte do modelo ou ``None`` se ele não tiver um."""
    modelo = onnx.load(str(caminho_onnx), load_external_data=False)
    valores = [prop.value for prop in modelo.metadata_props if prop.key == CHAVE]
    return json.loads(valores[-1]) if valores else None  # a última ocorrência vence, como no leitor web


def carimbar(
    caminho_onnx: str | Path, passaporte: dict, destino: str | Path | None = None
) -> Path:
    """Valida e grava o passaporte no modelo. Substitui um passaporte anterior."""
    caminho_onnx = Path(caminho_onnx)
    destino = Path(destino) if destino else caminho_onnx
    modelo = onnx.load(str(caminho_onnx))
    validar(passaporte, modelo)

    antigas = [p for p in modelo.metadata_props if p.key != CHAVE]
    del modelo.metadata_props[:]
    modelo.metadata_props.extend(antigas)
    prop = modelo.metadata_props.add()
    prop.key = CHAVE
    prop.value = json.dumps(passaporte, ensure_ascii=False)

    onnx.checker.check_model(modelo)
    onnx.save(modelo, str(destino))
    return destino


def _resumo(passaporte: dict) -> str:
    linhas = [
        f"Nome:     {passaporte['nome']}",
        f"Tarefa:   {passaporte['tarefa']}",
        f"Classes:  {len(passaporte['classes'])} -> {', '.join(passaporte['classes'])}",
        f"Entrada:  {passaporte['entrada']['tensor']} {passaporte['entrada']['layout']} "
        f"{passaporte['entrada']['tamanho']} {passaporte['entrada']['redimensionar']}",
        f"Saída:    {passaporte['saida']['tensor']} ({passaporte['saida']['formato']})",
        f"Limiar:   {passaporte['operacao']['confianca_minima']}",
    ]
    if "decisao" in passaporte:
        d = passaporte["decisao"]
        linhas.append(f"Decisão:  alerta se {d['alertar_se']} >= {d['limiar_alerta']}")
    best = passaporte.get("origem", {}).get("best")
    if best:
        linhas.append(
            "Best:     " + ", ".join(f"{k}={v}" for k, v in best.items())
        )
    return "\n".join(linhas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="comando", required=True)

    p_mostrar = sub.add_parser("mostrar", help="exibe o passaporte de um .onnx")
    p_mostrar.add_argument("onnx")
    p_mostrar.add_argument("--json", action="store_true", help="imprime o JSON completo")

    p_gravar = sub.add_parser("gravar", help="grava um passaporte JSON em um .onnx")
    p_gravar.add_argument("onnx")
    p_gravar.add_argument("passaporte")
    p_gravar.add_argument("--destino")

    args = parser.parse_args(argv)

    if args.comando == "mostrar":
        passaporte = ler(args.onnx)
        if passaporte is None:
            print(f"{args.onnx} não tem passaporte ({CHAVE}).")
            return 1
        print(json.dumps(passaporte, ensure_ascii=False, indent=2) if args.json
              else _resumo(passaporte))
        return 0

    passaporte = json.loads(Path(args.passaporte).read_text(encoding="utf-8"))
    try:
        destino = carimbar(args.onnx, passaporte, args.destino)
    except PassaporteInvalido as erro:
        print(f"Passaporte inválido: {erro}", file=sys.stderr)
        return 2
    print(f"Passaporte gravado em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
