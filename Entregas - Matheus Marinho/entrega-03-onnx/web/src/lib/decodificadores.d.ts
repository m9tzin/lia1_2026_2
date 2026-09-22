import type { Decisao, Deteccao, Geometria, Passaporte, Resultado, SaidaBruta } from "./tipos"

export function geometria(w: number, h: number, entrada: Pick<Passaporte["entrada"], "tamanho" | "redimensionar">): Geometria
export function paraTensor(rgba: ArrayLike<number>, entrada: Passaporte["entrada"]): { dados: Float32Array; dims: number[] }
export function desfazer(caixa: number[], geo: Pick<Geometria, "dx" | "dy" | "sx" | "sy">, w: number, h: number): [number, number, number, number]
export function iou(a: number[], b: number[]): number
export function nms<T extends Pick<Deteccao, "indice" | "confianca" | "caixa">>(deteccoes: T[], limiarIou: number): T[]
export function softmax(valores: number[]): number[]
export function pisoDeDecodificacao(passaporte: Passaporte, confianca: number): number
export function decodificar(
  saida: SaidaBruta | { dados: Float32Array },
  passaporte: Passaporte,
  geo: Geometria | null,
  w: number,
  h: number,
  opcoes: { confianca: number; iou?: number },
): Resultado
export function decidir(resultado: Resultado, passaporte: Passaporte, confianca: number): Decisao
