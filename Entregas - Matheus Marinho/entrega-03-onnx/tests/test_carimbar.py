import copy

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from passaporte.carimbar import CHAVE, PassaporteInvalido, carimbar, ler, validar
from passaporte.de_pytorch import TAMANHO, exportar, passaporte_cnn


@pytest.fixture
def cnn(tmp_path):
    return exportar(tmp_path / "cnn.onnx")


def test_ler_devolve_o_passaporte_gravado(cnn):
    lido = ler(cnn)
    assert lido == passaporte_cnn()
    assert lido["classes"] == ["gato", "cachorro"]


def test_modelo_sem_passaporte_devolve_none(tmp_path, cnn):
    modelo = onnx.load(str(cnn))
    del modelo.metadata_props[:]
    limpo = tmp_path / "limpo.onnx"
    onnx.save(modelo, str(limpo))
    assert ler(limpo) is None


def test_carimbar_de_novo_substitui_sem_duplicar(cnn):
    novo = copy.deepcopy(ler(cnn))
    novo["nome"] = "outro nome"
    carimbar(cnn, novo)
    chaves = [p.key for p in onnx.load(str(cnn)).metadata_props]
    assert chaves.count(CHAVE) == 1
    assert ler(cnn)["nome"] == "outro nome"


def test_carimbo_nao_altera_a_inferencia(tmp_path, cnn):
    modelo = onnx.load(str(cnn))
    del modelo.metadata_props[:]
    limpo = tmp_path / "limpo.onnx"
    onnx.save(modelo, str(limpo))

    x = np.random.default_rng(0).random((1, 3, TAMANHO, TAMANHO), dtype=np.float32)
    antes = ort.InferenceSession(str(limpo)).run(None, {"imagem": x})[0]
    depois = ort.InferenceSession(str(cnn)).run(None, {"imagem": x})[0]
    np.testing.assert_array_equal(antes, depois)
    onnx.checker.check_model(onnx.load(str(cnn)))


@pytest.mark.parametrize(
    ("alterar", "trecho"),
    [
        (lambda p: p.pop("classes"), "classes"),
        (lambda p: p.update(tarefa="segmentacao"), "tarefa"),
        (lambda p: p["entrada"].update(layout="CHW"), "entrada/layout"),
        (lambda p: p.update(classes=["gato", "gato"]), "repetidos"),
        (lambda p: p.update(decisao={"alertar_se": ["lobo"], "limiar_alerta": 0.5,
                                     "mensagem": "x"}), "alertar_se"),
        (lambda p: p["saida"].update(formato="caixas_xywh_por_classe"), "exige tarefa deteccao"),
    ],
)
def test_esquema_rejeita_passaporte_invalido(alterar, trecho):
    p = passaporte_cnn()
    alterar(p)
    with pytest.raises(PassaporteInvalido, match=trecho):
        validar(p)


def test_classes_divergentes_do_grafo_sao_rejeitadas(cnn):
    p = ler(cnn)
    p["classes"] = ["gato", "cachorro", "coelho"]
    with pytest.raises(PassaporteInvalido, match="3 declaradas, mas a saída tem 2"):
        carimbar(cnn, p)


def test_tensor_inexistente_e_rejeitado(cnn):
    p = ler(cnn)
    p["entrada"]["tensor"] = "images"
    with pytest.raises(PassaporteInvalido, match="'images' não existe"):
        carimbar(cnn, p)


def test_tamanho_divergente_do_grafo_e_rejeitado(cnn):
    p = ler(cnn)
    p["entrada"]["tamanho"] = [224, 224]
    with pytest.raises(PassaporteInvalido, match="altura declarada 224"):
        carimbar(cnn, p)
