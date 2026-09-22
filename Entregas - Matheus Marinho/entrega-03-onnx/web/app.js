import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";

import { carimbar, inspecionar, problemas } from "./passaporte.js";
import { decidir, decodificar, geometria, paraTensor } from "./decodificadores.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const $ = (id) => document.getElementById(id);
const estado = {
  bytes: null, nomeArquivo: "", info: null, passaporte: null, sessao: null,
  imagem: null, saida: null, geo: null, ms: 0,
};

// ---------- modelo ----------

async function carregarModelo(arquivo) {
  estado.nomeArquivo = arquivo.name;
  estado.bytes = new Uint8Array(await arquivo.arrayBuffer());
  estado.sessao = null;
  estado.saida = null;
  $("rotuloModelo").textContent = `${arquivo.name} (${(arquivo.size / 1e6).toFixed(1)} MB)`;
  mensagemModelo("");

  try {
    estado.info = inspecionar(estado.bytes);
  } catch (e) {
    mensagemModelo(`Não é um ONNX válido: ${e.message}`, "erro");
    return;
  }
  const { passaporte, erroPassaporte } = estado.info;
  if (!passaporte) {
    mostrarCarimbo(erroPassaporte);
    return;
  }
  const erros = problemas(passaporte, estado.info);
  if (erros.length) {
    mostrarCarimbo(`Passaporte com problemas: ${erros.join("; ")}`, passaporte);
    return;
  }
  await ativar(passaporte);
}

async function ativar(passaporte) {
  estado.passaporte = passaporte;
  $("painelCarimbo").classList.add("oculto");
  mostrarPassaporte(passaporte);
  mensagemModelo("Criando sessão do ONNX Runtime…");
  try {
    estado.sessao = await ort.InferenceSession.create(estado.bytes, { executionProviders: ["wasm"] });
  } catch (e) {
    mensagemModelo(`ONNX Runtime recusou o modelo: ${e.message}`, "erro");
    return;
  }
  mensagemModelo("");
  const op = passaporte.operacao;
  $("confianca").value = op.confianca_minima;
  $("iou").value = op.iou_nms ?? 0.45;
  $("controleIou").classList.toggle("oculto", passaporte.tarefa !== "deteccao");
  $("painelControles").classList.remove("oculto");
  atualizarRotulosControles();
  if (estado.imagem) await inferir();
}

function mensagemModelo(texto, classe = "") {
  const el = $("estadoModelo");
  el.textContent = texto;
  el.className = texto ? classe : "oculto";
}

function linha(dl, rotulo, conteudo) {
  const dt = document.createElement("dt");
  dt.textContent = rotulo;
  const dd = document.createElement("dd");
  if (conteudo instanceof Node) dd.append(conteudo);
  else dd.textContent = conteudo;
  dl.append(dt, dd);
}

function mostrarPassaporte(p) {
  const dl = $("fichaPassaporte");
  dl.replaceChildren();
  linha(dl, "Nome", p.nome);
  linha(dl, "Tarefa", p.tarefa);
  const chips = document.createElement("div");
  chips.className = "classes";
  for (const c of p.classes.slice(0, 40)) {
    const s = document.createElement("span");
    s.textContent = c;
    chips.append(s);
  }
  if (p.classes.length > 40) chips.append(`+${p.classes.length - 40}`);
  linha(dl, `Classes (${p.classes.length})`, chips);
  const e = p.entrada;
  linha(dl, "Entrada", `${e.tensor} ${e.layout} ${e.tamanho.join("×")} ${e.cores}, ${e.redimensionar}`);
  linha(dl, "Saída", `${p.saida.tensor}, ${p.saida.formato}`);
  if (p.decisao) {
    const d = p.decisao;
    const zona = d.zona_incerta ? `; verificar entre ${d.zona_incerta.join(" e ")}` : "";
    linha(dl, "Decisão", `${d.mensagem}: ${d.alertar_se.join(", ")} ≥ ${d.limiar_alerta}${zona}`);
  }
  const o = p.origem ?? {};
  if (o.framework) linha(dl, "Origem", [o.framework, o.pesos, o.dataset].filter(Boolean).join(" · "));
  if (o.best) {
    const b = o.best;
    const partes = [];
    if (b.epoca != null) partes.push(`época ${b.epoca}${b.epocas_treinadas ? ` de ${b.epocas_treinadas}` : ""}`);
    if (b.mAP50 != null) partes.push(`mAP50 ${b.mAP50}`);
    if (b.mAP50_95 != null) partes.push(`mAP50-95 ${b.mAP50_95}`);
    linha(dl, "Best", partes.join(" · "));
  }
  if (o.exportado_em) linha(dl, "Exportado", o.exportado_em);
  $("painelPassaporte").classList.remove("oculto");
}

// ---------- fallback: carimbar no navegador ----------

// Nomes que o Ultralytics grava como dict Python: {0: 'person', 1: "o'clock"}.
function nomesUltralytics(metadados) {
  const bruto = metadados.names;
  if (!bruto) return null;
  const nomes = [];
  for (const m of bruto.matchAll(/(\d+)\s*:\s*(['"])(.*?)\2\s*[,}]/g)) nomes[Number(m[1])] = m[3];
  return nomes.length ? nomes : null;
}

function rascunho(info) {
  const ent = info.entradas[0] ?? { nome: "entrada", dims: [] };
  const sai = info.saidas[0] ?? { nome: "saida", dims: [] };
  const d = ent.dims;
  const nhwc = d.length === 4 && d[3] === 3;
  const tamanho = nhwc ? [d[1], d[2]] : [d[2], d[3]];
  const estatico = tamanho.every((v) => typeof v === "number" && v > 0);

  // Sugestão a partir da shape de saída; o usuário confirma antes de carimbar.
  const s = sai.dims;
  const classificacao = s.length === 2;
  let nc = classificacao ? s[1] : (typeof s[1] === "number" ? s[1] - 4 : 1);
  const doUltralytics = nomesUltralytics(info.metadados);
  if (doUltralytics) nc = doUltralytics.length;
  const classes = doUltralytics ?? Array.from({ length: Math.max(nc, 1) }, (_, i) => `classe_${i}`);

  return {
    versao: 1,
    nome: estado.nomeArquivo.replace(/\.onnx$/, ""),
    tarefa: classificacao ? "classificacao" : "deteccao",
    classes,
    entrada: {
      tensor: ent.nome, layout: nhwc ? "NHWC" : "NCHW", tamanho: estatico ? tamanho : [640, 640],
      cores: "RGB", redimensionar: classificacao ? "esticar" : "letterbox", preenchimento: 114,
      escala: 1 / 255, media: [0, 0, 0], desvio: [1, 1, 1],
    },
    saida: { tensor: sai.nome, formato: classificacao ? "logits" : "caixas_xywh_por_classe" },
    operacao: { confianca_minima: 0.25, iou_nms: 0.45, rotulo_incerto: "INCERTO" },
    origem: { framework: info.produtor || "desconhecido", pesos: estado.nomeArquivo },
  };
}

function mostrarCarimbo(motivo, passaporte = null) {
  $("painelPassaporte").classList.add("oculto");
  $("painelControles").classList.add("oculto");
  const tensores = [
    ...estado.info.entradas.map((t) => `entrada ${t.nome} [${t.dims.join(", ")}]`),
    ...estado.info.saidas.map((t) => `saída ${t.nome} [${t.dims.join(", ")}]`),
  ].join(" · ");
  mensagemModelo(motivo ? `${motivo}. Grafo: ${tensores}` : `Grafo: ${tensores}`, motivo ? "aviso" : "");
  $("rascunho").value = JSON.stringify(passaporte ?? rascunho(estado.info), null, 2);
  $("errosRascunho").replaceChildren();
  $("painelCarimbo").classList.remove("oculto");
}

async function carimbarRascunho() {
  const lista = $("errosRascunho");
  lista.replaceChildren();
  let p;
  try {
    p = JSON.parse($("rascunho").value);
  } catch (e) {
    lista.append(Object.assign(document.createElement("li"), { textContent: `JSON inválido: ${e.message}` }));
    return;
  }
  const erros = problemas(p, estado.info);
  if (erros.length) {
    for (const e of erros) lista.append(Object.assign(document.createElement("li"), { textContent: e }));
    return;
  }
  estado.bytes = carimbar(estado.bytes, p);
  estado.info = inspecionar(estado.bytes);
  const nome = estado.nomeArquivo.replace(/\.onnx$/, "") + "_carimbado.onnx";
  const url = URL.createObjectURL(new Blob([estado.bytes], { type: "application/octet-stream" }));
  Object.assign(document.createElement("a"), { href: url, download: nome }).click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  await ativar(estado.info.passaporte);
}

// ---------- imagem e inferência ----------

async function carregarImagem(arquivo) {
  const img = new Image();
  img.src = URL.createObjectURL(arquivo);
  await img.decode();
  estado.imagem = img;
  if (estado.sessao) await inferir();
  else desenhar();
}

async function inferir() {
  const p = estado.passaporte;
  const img = estado.imagem;
  const e = p.entrada;
  const [H, W] = e.tamanho;
  const geo = geometria(img.naturalWidth, img.naturalHeight, e);

  const c = new OffscreenCanvas(W, H);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const pad = e.preenchimento ?? 114;
  ctx.fillStyle = `rgb(${pad}, ${pad}, ${pad})`;
  ctx.fillRect(0, 0, W, H);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, geo.dx, geo.dy, geo.dw, geo.dh);
  const { dados, dims } = paraTensor(ctx.getImageData(0, 0, W, H).data, e);

  const t0 = performance.now();
  const resultados = await estado.sessao.run({ [e.tensor]: new ort.Tensor("float32", dados, dims) });
  estado.ms = performance.now() - t0;
  const saida = resultados[p.saida.tensor];
  estado.saida = { dados: saida.data, dims: saida.dims };
  estado.geo = geo;
  desenhar();
}

function desenhar() {
  const img = estado.imagem;
  if (!img) return;
  const tela = $("tela");
  tela.width = img.naturalWidth;
  tela.height = img.naturalHeight;
  const ctx = tela.getContext("2d");
  ctx.drawImage(img, 0, 0);
  tela.classList.remove("oculto");
  $("vazio").classList.add("oculto");
  if (!estado.saida) return;

  const p = estado.passaporte;
  const confianca = Number($("confianca").value);
  const r = decodificar(estado.saida, p, estado.geo, img.naturalWidth, img.naturalHeight, {
    confianca, iou: Number($("iou").value),
  });
  const decisao = decidir(r, p, confianca);

  $("seloEstado").textContent = decisao.estado;
  $("seloMotivo").textContent = decisao.motivo;
  $("selo").className = `selo ${decisao.estado}`;

  const tbody = $("tabela").querySelector("tbody");
  tbody.replaceChildren();
  const cores = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c"];
  const alerta = new Set(p.decisao?.alertar_se ?? []);

  if (r.tipo === "deteccao") {
    $("colCaixa").classList.remove("oculto");
    const espessura = Math.max(2, Math.round(Math.max(tela.width, tela.height) / 300));
    ctx.font = `${espessura * 7}px system-ui, sans-serif`;
    ctx.textBaseline = "top";
    const mostrar = r.deteccoes.length ? r.deteccoes : [];
    // Candidata na zona incerta aparece tracejada, mesmo abaixo do limiar de exibição.
    const extra = decisao.estado === "VERIFICAR" && decisao.alvo ? [{ ...decisao.alvo, tracejada: true }] : [];
    for (const d of [...mostrar, ...extra]) {
      const [x1, y1, x2, y2] = d.caixa;
      const cor = alerta.has(d.classe) ? "#d32f2f" : cores[d.indice % cores.length];
      ctx.strokeStyle = cor;
      ctx.lineWidth = espessura;
      ctx.setLineDash(d.tracejada ? [espessura * 4, espessura * 3] : []);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const rotulo = `${d.classe} ${Math.round(d.confianca * 100)}%`;
      const largura = ctx.measureText(rotulo).width + espessura * 4;
      const altura = espessura * 9;
      const ty = y1 - altura >= 0 ? y1 - altura : y1;
      ctx.setLineDash([]);
      ctx.fillStyle = cor;
      ctx.fillRect(x1, ty, largura, altura);
      ctx.fillStyle = "#fff";
      ctx.fillText(rotulo, x1 + espessura * 2, ty + espessura);
    }
    const linhas = r.deteccoes.length ? r.deteccoes : r.candidatas.slice(0, 1);
    for (const d of linhas) {
      const tr = tbody.insertRow();
      tr.insertCell().textContent = d.classe + (r.deteccoes.length ? "" : " (melhor candidata, abaixo do limiar)");
      Object.assign(tr.insertCell(), { className: "num", textContent: `${(d.confianca * 100).toFixed(1)}%` });
      tr.insertCell().textContent = d.caixa.map((v) => Math.round(v)).join(", ");
    }
  } else {
    $("colCaixa").classList.add("oculto");
    for (const d of r.ranking.slice(0, 3)) {
      const tr = tbody.insertRow();
      tr.insertCell().textContent = d.classe;
      Object.assign(tr.insertCell(), { className: "num", textContent: `${(d.confianca * 100).toFixed(1)}%` });
    }
  }
  $("tabela").classList.toggle("oculto", tbody.rows.length === 0);
  $("tempo").textContent = `Inferência: ${estado.ms.toFixed(0)} ms (wasm, CPU)`;
  $("tempo").classList.remove("oculto");
}

function atualizarRotulosControles() {
  $("valorConfianca").textContent = `${Math.round($("confianca").value * 100)}%`;
  $("valorIou").textContent = Number($("iou").value).toFixed(2);
}

// ---------- eventos ----------

function zonaDeSoltar(zona, input, aoReceber) {
  input.addEventListener("change", () => input.files[0] && aoReceber(input.files[0]));
  zona.addEventListener("dragover", (e) => {
    e.preventDefault();
    zona.classList.add("ativo");
  });
  zona.addEventListener("dragleave", () => zona.classList.remove("ativo"));
  zona.addEventListener("drop", (e) => {
    e.preventDefault();
    zona.classList.remove("ativo");
    const arquivo = e.dataTransfer.files[0];
    if (arquivo) aoReceber(arquivo);
  });
}

zonaDeSoltar($("soltarModelo"), $("arquivoModelo"), carregarModelo);
zonaDeSoltar($("soltarImagem"), $("arquivoImagem"), carregarImagem);
$("botaoCarimbar").addEventListener("click", carimbarRascunho);
for (const id of ["confianca", "iou"]) {
  $(id).addEventListener("input", () => {
    atualizarRotulosControles();
    desenhar();
  });
}
