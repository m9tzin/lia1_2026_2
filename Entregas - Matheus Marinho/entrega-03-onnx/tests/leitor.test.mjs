// node --test tests/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { carimbar, inspecionar, problemas } from "../web/src/lib/passaporte.js";
import {
  decidir,
  decodificar,
  desfazer,
  geometria,
  nms,
  paraTensor,
  softmax,
} from "../web/src/lib/decodificadores.js";

const fixture = (nome) => readFileSync(new URL(`./fixtures/${nome}`, import.meta.url));

test("lê o passaporte gravado pelo Python", () => {
  const info = inspecionar(fixture("cnn.onnx"));
  assert.equal(info.passaporte.nome, "CNN gatos e cachorros (PyTorch)");
  assert.deepEqual(info.passaporte.classes, ["gato", "cachorro"]);
  assert.deepEqual(info.entradas, [{ nome: "imagem", dims: [1, 3, 128, 128] }]);
  assert.deepEqual(info.saidas, [{ nome: "logits", dims: [1, 2] }]);
  assert.equal(info.produtor, "pytorch");
  assert.deepEqual(problemas(info.passaporte, info), []);
});

test("modelo sem passaporte é reconhecido como tal, com o grafo disponível", () => {
  const info = inspecionar(fixture("cnn_sem_passaporte.onnx"));
  assert.equal(info.passaporte, null);
  assert.equal(info.entradas[0].nome, "imagem");
});

test("carimbar no navegador grava, relê e substitui sem duplicar a chave", () => {
  const original = fixture("cnn_sem_passaporte.onnx");
  const p = inspecionar(fixture("cnn.onnx")).passaporte;
  const uma = carimbar(original, { ...p, nome: "primeiro" });
  const duas = carimbar(uma, { ...p, nome: "segundo" });
  assert.equal(inspecionar(uma).passaporte.nome, "primeiro");
  assert.equal(inspecionar(duas).passaporte.nome, "segundo");
  // tamanho só cresce pelo passaporte: o antigo foi removido, não acumulado
  assert.ok(duas.length - original.length < 2 * JSON.stringify(p).length);
  assert.deepEqual(inspecionar(duas).entradas, inspecionar(original).entradas);
});

test("problemas aponta tensor inexistente e classe de alerta desconhecida", () => {
  const info = inspecionar(fixture("cnn.onnx"));
  const p = structuredClone(info.passaporte);
  p.entrada.tensor = "images";
  p.decisao = { alertar_se: ["lobo"], limiar_alerta: 0.5, mensagem: "x" };
  const erros = problemas(p, info);
  assert.ok(erros.some((e) => e.includes("'images'")));
  assert.ok(erros.some((e) => e.includes("lobo")));
});

const letterbox = { tamanho: [640, 640], redimensionar: "letterbox" };

test("letterbox centraliza e desfazer volta para a imagem original", () => {
  const geo = geometria(810, 1080, letterbox); // bus.jpg
  assert.equal(geo.dh, 640);
  assert.equal(geo.dw, 480);
  assert.equal(geo.dx, 80);
  assert.equal(geo.dy, 0);
  const caixa = [80 + 48, 64, 80 + 96, 128]; // no tensor
  const [x1, y1, x2, y2] = desfazer(caixa, geo, 810, 1080);
  assert.ok(Math.abs(x1 - 81) < 0.5 && Math.abs(x2 - 162) < 0.5);
  assert.ok(Math.abs(y1 - 108) < 0.5 && Math.abs(y2 - 216) < 0.5);
});

test("recorte central cobre o tensor inteiro", () => {
  const geo = geometria(400, 200, { tamanho: [100, 100], redimensionar: "recorte_central" });
  assert.equal(geo.dh, 100);
  assert.equal(geo.dw, 200);
  assert.equal(geo.dx, -50);
});

test("paraTensor normaliza e troca o layout", () => {
  const rgba = Uint8Array.from([255, 0, 0, 255, 0, 255, 0, 255]); // 2 pixels
  const entrada = { tamanho: [1, 2], layout: "NCHW", cores: "RGB", escala: 1 / 255, media: [0, 0, 0], desvio: [1, 1, 1] };
  const { dados, dims } = paraTensor(rgba, entrada);
  assert.deepEqual(dims, [1, 3, 1, 2]);
  assert.deepEqual(Array.from(dados), [1, 0, 0, 1, 0, 0]);
  const nhwc = paraTensor(rgba, { ...entrada, layout: "NHWC", cores: "BGR" });
  assert.deepEqual(Array.from(nhwc.dados), [0, 0, 1, 0, 1, 0]);
});

test("nms suprime por classe, não entre classes", () => {
  const a = { indice: 0, confianca: 0.9, caixa: [0, 0, 10, 10] };
  const b = { indice: 0, confianca: 0.8, caixa: [1, 1, 10, 10] };
  const c = { indice: 1, confianca: 0.7, caixa: [0, 0, 10, 10] };
  assert.deepEqual(nms([b, c, a], 0.45), [a, c]);
});

// Passaporte de detecção com 2 classes e uma saída [1, 6, N] montada à mão.
const detector = {
  classes: ["capacete", "sem capacete"],
  saida: { formato: "caixas_xywh_por_classe" },
  operacao: { confianca_minima: 0.5 },
  decisao: { alertar_se: ["sem capacete"], limiar_alerta: 0.5, zona_incerta: [0.3, 0.5], mensagem: "Sem capacete" },
};

function saidaYolo(ancoras) {
  const N = ancoras.length;
  const dados = new Float32Array(6 * N);
  ancoras.forEach((a, i) => a.forEach((v, canal) => (dados[canal * N + i] = v)));
  return { dados, dims: [1, 6, N] };
}

const semBorda = { dx: 0, dy: 0, sx: 1, sy: 1 };

test("decodifica caixas YOLO e aplica a decisão ALERTA", () => {
  const saida = saidaYolo([
    [50, 50, 20, 20, 0.9, 0.1],
    [52, 50, 20, 20, 0.8, 0.1], // sobrepõe a primeira, some no NMS
    [200, 200, 30, 30, 0.1, 0.7],
  ]);
  const r = decodificar(saida, detector, semBorda, 640, 640, { confianca: 0.5, iou: 0.45 });
  assert.equal(r.deteccoes.length, 2);
  assert.deepEqual(r.deteccoes.map((d) => d.classe), ["capacete", "sem capacete"]);
  assert.deepEqual(r.deteccoes[0].caixa, [40, 40, 60, 60]);
  assert.equal(decidir(r, detector, 0.5).estado, "ALERTA");
});

test("candidata na zona incerta vira VERIFICAR mesmo abaixo do limiar de exibição", () => {
  const saida = saidaYolo([[200, 200, 30, 30, 0.05, 0.4]]);
  const r = decodificar(saida, detector, semBorda, 640, 640, { confianca: 0.5, iou: 0.45 });
  assert.equal(r.deteccoes.length, 0);
  assert.equal(r.candidatas.length, 1);
  assert.equal(decidir(r, detector, 0.5).estado, "VERIFICAR");
});

test("sem nada relevante a decisão é OK", () => {
  const saida = saidaYolo([[50, 50, 20, 20, 0.9, 0.1]]);
  const r = decodificar(saida, detector, semBorda, 640, 640, { confianca: 0.5, iou: 0.45 });
  assert.equal(decidir(r, detector, 0.5).estado, "OK");
});

test("formato end2end [1, K, 6] é lido sem NMS", () => {
  const p = { ...detector, saida: { formato: "caixas_xyxy_conf_classe" } };
  const dados = Float32Array.from([10, 10, 20, 20, 0.9, 1, 0, 0, 0, 0, 0.0, 0]);
  const r = decodificar({ dados, dims: [1, 2, 6] }, p, semBorda, 640, 640, { confianca: 0.5, iou: 0.45 });
  assert.equal(r.deteccoes.length, 1);
  assert.equal(r.deteccoes[0].classe, "sem capacete");
});

test("classificação: logits passam por softmax e top 1 fraco é INCERTO", () => {
  const p = { classes: ["gato", "cachorro"], saida: { formato: "logits" }, operacao: { rotulo_incerto: "INCERTO" } };
  const r = decodificar({ dados: Float32Array.from([0.1, 0.3]) }, p, null, 0, 0, { confianca: 0.7 });
  assert.equal(r.ranking[0].classe, "cachorro");
  assert.ok(Math.abs(r.ranking[0].confianca + r.ranking[1].confianca - 1) < 1e-6);
  assert.equal(decidir(r, p, 0.7).estado, "INCERTO");
  const forte = decodificar({ dados: Float32Array.from([5, 0]) }, p, null, 0, 0, { confianca: 0.7 });
  assert.equal(decidir(forte, p, 0.7).estado, "gato");
});

test("softmax é estável com valores grandes", () => {
  const s = softmax([1000, 1000]);
  assert.deepEqual(s, [0.5, 0.5]);
});
