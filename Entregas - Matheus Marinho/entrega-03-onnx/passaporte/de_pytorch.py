"""Exporta uma CNN PyTorch comum para ONNX e carimba um passaporte escrito à mão.

Serve para provar que o leitor não depende do Ultralytics: o grafo aqui não tem nenhum metadado
de YOLO, só o que o passaporte declara. A rede segue o estilo da CNN de gatos e cachorros da Aula 12.
Sem ``--pesos`` ela sai com pesos aleatórios, o que basta para testar o contrato de ponta a ponta.

Uso:
    python -m passaporte.de_pytorch --destino modelos/cnn_gatos_cachorros.onnx
    python -m passaporte.de_pytorch --pesos cnn.pth --destino modelos/cnn.onnx
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import torch
from torch import nn

from passaporte.carimbar import carimbar

TAMANHO = 128
CLASSES = ["gato", "cachorro"]


class CNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def passaporte_cnn() -> dict:
    return {
        "versao": 1,
        "nome": "CNN gatos e cachorros (PyTorch)",
        "tarefa": "classificacao",
        "classes": CLASSES,
        "entrada": {
            "tensor": "imagem",
            "layout": "NCHW",
            "tamanho": [TAMANHO, TAMANHO],
            "cores": "RGB",
            "redimensionar": "esticar",
            "escala": 1 / 255,
            "media": [0.485, 0.456, 0.406],
            "desvio": [0.229, 0.224, 0.225],
        },
        "saida": {"tensor": "logits", "formato": "logits"},
        "operacao": {"confianca_minima": 0.7, "rotulo_incerto": "INCERTO"},
        "origem": {
            "framework": f"pytorch {torch.__version__}",
            "pesos": "aleatorios",
            "exportado_em": date.today().isoformat(),
        },
    }


def exportar(destino: str | Path, pesos: str | Path | None = None) -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    modelo = CNN().eval()
    passaporte = passaporte_cnn()
    if pesos:
        modelo.load_state_dict(torch.load(pesos, map_location="cpu"))
        passaporte["origem"]["pesos"] = Path(pesos).name

    exemplo = torch.zeros(1, 3, TAMANHO, TAMANHO)
    torch.onnx.export(
        modelo, exemplo, str(destino),
        input_names=["imagem"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )
    carimbar(destino, passaporte)
    return destino


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--destino", default="modelos/cnn_gatos_cachorros.onnx")
    parser.add_argument("--pesos", help="state_dict .pth treinado (opcional)")
    args = parser.parse_args(argv)
    print(f"ONNX carimbado: {exportar(args.destino, args.pesos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
