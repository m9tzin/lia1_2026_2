// Ponte entre o passaporte e o ONNX Runtime Web: cria a sessão, prepara a imagem e roda.
import * as ort from "onnxruntime-web/wasm"

import { geometria, paraTensor } from "./decodificadores.js"
import { carimbar as carimbarBytes, inspecionar as inspecionarBytes, problemas as problemasJs } from "./passaporte.js"
import type { Geometria, InfoModelo, Passaporte, SaidaBruta } from "./tipos"

export const inspecionar = (bytes: Uint8Array): InfoModelo => inspecionarBytes(bytes)
export const problemas = (p: unknown, info?: InfoModelo | null): string[] => problemasJs(p, info ?? null)
export const carimbar = (bytes: Uint8Array, p: Passaporte): Uint8Array => carimbarBytes(bytes, p)

export type Sessao = ort.InferenceSession

export function criarSessao(bytes: Uint8Array): Promise<Sessao> {
  return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] })
}

export interface Execucao {
  saida: SaidaBruta
  geo: Geometria
  ms: number
}

// Desenha a imagem no tamanho de entrada conforme o passaporte, converte e roda a sessão.
export async function executar(sessao: Sessao, p: Passaporte, img: HTMLImageElement): Promise<Execucao> {
  const e = p.entrada
  const [H, W] = e.tamanho
  const geo: Geometria = geometria(img.naturalWidth, img.naturalHeight, e)

  const tela = new OffscreenCanvas(W, H)
  const ctx = tela.getContext("2d", { willReadFrequently: true })!
  const pad = e.preenchimento ?? 114
  ctx.fillStyle = `rgb(${pad}, ${pad}, ${pad})`
  ctx.fillRect(0, 0, W, H)
  ctx.imageSmoothingQuality = "high"
  ctx.drawImage(img, geo.dx, geo.dy, geo.dw, geo.dh)
  const { dados, dims } = paraTensor(ctx.getImageData(0, 0, W, H).data, e)

  const t0 = performance.now()
  const resultados = await sessao.run({ [e.tensor]: new ort.Tensor("float32", dados, dims) })
  const ms = performance.now() - t0
  const t = resultados[p.saida.tensor]
  return { saida: { dados: t.data as Float32Array, dims: t.dims }, geo, ms }
}

// Nomes que o Ultralytics grava como dict Python: {0: 'person', 1: "o'clock"}.
function nomesUltralytics(metadados: Record<string, string>): string[] | null {
  const bruto = metadados.names
  if (!bruto) return null
  const nomes: string[] = []
  for (const m of bruto.matchAll(/(\d+)\s*:\s*(['"])(.*?)\2\s*[,}]/g)) nomes[Number(m[1])] = m[3]
  return nomes.length ? nomes : null
}

// Rascunho de passaporte para um .onnx sem contrato: sugestões lidas do grafo, o usuário confirma.
export function rascunho(info: InfoModelo, nomeArquivo: string): Passaporte {
  const ent = info.entradas[0] ?? { nome: "entrada", dims: [] }
  const sai = info.saidas[0] ?? { nome: "saida", dims: [] }
  const d = ent.dims
  const nhwc = d.length === 4 && d[3] === 3
  const tamanho = (nhwc ? [d[1], d[2]] : [d[2], d[3]]) as (number | null)[]
  const estatico = tamanho.every((v) => typeof v === "number" && v > 0)

  const s = sai.dims
  const classificacao = s.length === 2
  let nc = classificacao ? Number(s[1]) : typeof s[1] === "number" ? s[1] - 4 : 1
  const doUltralytics = nomesUltralytics(info.metadados)
  if (doUltralytics) nc = doUltralytics.length
  const classes = doUltralytics ?? Array.from({ length: Math.max(nc || 1, 1) }, (_, i) => `classe_${i}`)

  return {
    versao: 1,
    nome: nomeArquivo.replace(/\.onnx$/, ""),
    tarefa: classificacao ? "classificacao" : "deteccao",
    classes,
    entrada: {
      tensor: ent.nome,
      layout: nhwc ? "NHWC" : "NCHW",
      tamanho: (estatico ? tamanho : [640, 640]) as [number, number],
      cores: "RGB",
      redimensionar: classificacao ? "esticar" : "letterbox",
      preenchimento: 114,
      escala: 1 / 255,
      media: [0, 0, 0],
      desvio: [1, 1, 1],
    },
    saida: { tensor: sai.nome, formato: classificacao ? "logits" : "caixas_xywh_por_classe" },
    operacao: { confianca_minima: 0.25, iou_nms: 0.45, rotulo_incerto: "INCERTO" },
    origem: { framework: info.produtor || "desconhecido", pesos: nomeArquivo },
  }
}

export function baixar(bytes: Uint8Array, nome: string) {
  const url = URL.createObjectURL(new Blob([bytes as BlobPart], { type: "application/octet-stream" }))
  Object.assign(document.createElement("a"), { href: url, download: nome }).click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
