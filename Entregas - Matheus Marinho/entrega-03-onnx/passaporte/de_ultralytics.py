"""Leva o best.pt de um treino Ultralytics até um .onnx carimbado com passaporte.

O que é o best.pt: durante ``model.train(...)`` o Ultralytics valida o modelo ao fim de cada época
e calcula um *fitness*. Ele grava em ``runs/detect/<nome>/weights/`` o ``last.pt`` (última época) e
o ``best.pt`` (época de maior fitness). O best.pt guarda em ``train_metrics`` as métricas daquela
época, que este script copia para o passaporte.

Uso:
    python -m passaporte.de_ultralytics                          # best.pt mais recente em runs/
    python -m passaporte.de_ultralytics --pesos caminho/best.pt --classes classes_pt.txt \
        --nome "Capacete em obra" --dataset hard-hat-workers \
        --alertar-se "sem capacete" --limiar-alerta 0.5 --zona-incerta 0.3 0.5
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import date
from pathlib import Path

from passaporte.carimbar import PassaporteInvalido, carimbar

CHAVE_MAP50 = "metrics/mAP50(B)"
CHAVE_MAP50_95 = "metrics/mAP50-95(B)"


def encontrar_best(raiz: str | Path = "runs") -> Path:
    """Devolve o best.pt modificado mais recentemente dentro de ``raiz``."""
    candidatos = sorted(Path(raiz).glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidatos:
        raise FileNotFoundError(f"nenhum weights/best.pt encontrado em {Path(raiz).resolve()}")
    return candidatos[-1]


def ler_resultados(pasta_treino: Path) -> list[dict[str, float]]:
    """Lê o results.csv do treino (uma linha por época)."""
    arquivo = pasta_treino / "results.csv"
    if not arquivo.exists():
        return []
    with arquivo.open(newline="", encoding="utf-8") as f:
        return [
            {k.strip(): float(v) for k, v in linha.items() if v not in ("", None)}
            for linha in csv.DictReader(f)
        ]


def metricas_do_best(pesos: Path) -> dict:
    """Métricas da época que gerou o best.pt.

    Usa ``train_metrics`` salvo no próprio checkpoint e localiza a época correspondente no
    results.csv comparando o mAP50-95. Assim não depende da fórmula de fitness da versão instalada.
    """
    from ultralytics.utils.patches import torch_load

    ckpt = torch_load(pesos, map_location="cpu")
    metricas = ckpt.get("train_metrics") or {}
    if CHAVE_MAP50_95 not in metricas:
        return {}

    best = {
        "mAP50": round(float(metricas[CHAVE_MAP50]), 4),
        "mAP50_95": round(float(metricas[CHAVE_MAP50_95]), 4),
        "precisao": round(float(metricas.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(metricas.get("metrics/recall(B)", 0.0)), 4),
        "fitness": round(float(metricas.get("fitness", metricas[CHAVE_MAP50_95])), 4),
    }
    linhas = ler_resultados(pesos.parent.parent)
    if linhas:
        alvo = float(metricas[CHAVE_MAP50_95])
        linha = min(linhas, key=lambda l: abs(l.get(CHAVE_MAP50_95, -1.0) - alvo))
        best["epoca"] = int(linha["epoch"])
        best["epocas_treinadas"] = int(max(l["epoch"] for l in linhas))
    return best


def ler_classes(arquivo: str | Path) -> list[str]:
    """Uma classe por linha, na mesma ordem dos IDs do modelo."""
    linhas = Path(arquivo).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in linhas if l.strip()]


def montar_passaporte(
    *,
    classes: list[str],
    imgsz: int,
    end2end: bool,
    nome: str,
    dataset: str | None,
    versao_ultralytics: str,
    pesos: Path,
    best: dict,
    confianca_minima: float,
    decisao: dict | None,
) -> dict:
    passaporte = {
        "versao": 1,
        "nome": nome,
        "tarefa": "deteccao",
        "classes": classes,
        "entrada": {
            "tensor": "images",
            "layout": "NCHW",
            "tamanho": [imgsz, imgsz],
            "cores": "RGB",
            "redimensionar": "letterbox",
            "preenchimento": 114,
            "escala": 1 / 255,
            "media": [0, 0, 0],
            "desvio": [1, 1, 1],
        },
        "saida": {
            "tensor": "output0",
            # YOLO26 e YOLOv10 exportam caixas já filtradas; YOLOv8/11 exportam as âncoras cruas.
            "formato": "caixas_xyxy_conf_classe" if end2end else "caixas_xywh_por_classe",
        },
        "operacao": {"confianca_minima": confianca_minima, "iou_nms": 0.45},
        "origem": {
            "framework": f"ultralytics {versao_ultralytics}",
            "pesos": pesos.name,
            "exportado_em": date.today().isoformat(),
        },
    }
    if dataset:
        passaporte["origem"]["dataset"] = dataset
    if best:
        passaporte["origem"]["best"] = best
    if decisao:
        passaporte["decisao"] = decisao
    return passaporte


def exportar(
    pesos: str | Path,
    *,
    destino: str | Path | None = None,
    classes: list[str] | None = None,
    nome: str | None = None,
    dataset: str | None = None,
    imgsz: int = 640,
    confianca_minima: float = 0.25,
    decisao: dict | None = None,
) -> tuple[Path, dict]:
    """Exporta ``pesos`` para ONNX (parâmetros da Aula 13) e grava o passaporte."""
    import ultralytics
    from ultralytics import YOLO

    pesos = Path(pesos)
    modelo = YOLO(str(pesos))
    if modelo.task != "detect":
        raise ValueError(f"v1 do passaporte cobre só detecção; o modelo é '{modelo.task}'")

    nomes = [modelo.names[i] for i in range(len(modelo.names))]
    if classes is not None and len(classes) != len(nomes):
        raise PassaporteInvalido(
            f"--classes tem {len(classes)} nomes, o modelo tem {len(nomes)}: {nomes}"
        )
    end2end = bool(getattr(modelo.model, "end2end", False))

    caminho = Path(modelo.export(format="onnx", imgsz=imgsz, simplify=True, opset=17))
    if destino:
        destino = Path(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(caminho, destino)
        caminho = destino

    passaporte = montar_passaporte(
        classes=classes or nomes,
        imgsz=imgsz,
        end2end=end2end,
        nome=nome or pesos.parent.parent.name,
        dataset=dataset,
        versao_ultralytics=ultralytics.__version__,
        pesos=pesos,
        best=metricas_do_best(pesos),
        confianca_minima=confianca_minima,
        decisao=decisao,
    )
    carimbar(caminho, passaporte)
    return caminho, passaporte


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pesos", help="caminho do best.pt (padrão: o mais recente em --runs)")
    parser.add_argument("--runs", default="runs", help="pasta onde procurar o best.pt")
    parser.add_argument("--destino", help="caminho do .onnx de saída")
    parser.add_argument("--classes", help="txt com uma classe por linha (tradução)")
    parser.add_argument("--nome", help="nome legível do modelo")
    parser.add_argument("--dataset", help="nome do dataset usado no treino")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confianca", type=float, default=0.25)
    parser.add_argument("--alertar-se", nargs="+", metavar="CLASSE",
                        help="classes que disparam alerta")
    parser.add_argument("--limiar-alerta", type=float, default=0.5)
    parser.add_argument("--zona-incerta", type=float, nargs=2, metavar=("INICIO", "FIM"))
    parser.add_argument("--mensagem", default="Alerta")
    args = parser.parse_args(argv)

    pesos = Path(args.pesos) if args.pesos else encontrar_best(args.runs)
    print(f"best.pt: {pesos}")

    decisao = None
    if args.alertar_se:
        decisao = {
            "alertar_se": args.alertar_se,
            "limiar_alerta": args.limiar_alerta,
            "mensagem": args.mensagem,
        }
        if args.zona_incerta:
            decisao["zona_incerta"] = args.zona_incerta

    try:
        caminho, passaporte = exportar(
            pesos,
            destino=args.destino,
            classes=ler_classes(args.classes) if args.classes else None,
            nome=args.nome,
            dataset=args.dataset,
            imgsz=args.imgsz,
            confianca_minima=args.confianca,
            decisao=decisao,
        )
    except PassaporteInvalido as erro:
        print(f"Passaporte inválido: {erro}", file=sys.stderr)
        return 2

    best = passaporte["origem"].get("best", {})
    print(f"ONNX carimbado: {caminho}")
    print(f"Classes: {passaporte['classes']}")
    if best:
        print(f"Época do best: {best.get('epoca', '?')} | mAP50 {best['mAP50']} | "
              f"mAP50-95 {best['mAP50_95']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
