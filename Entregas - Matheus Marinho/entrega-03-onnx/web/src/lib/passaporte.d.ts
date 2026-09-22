import type { InfoModelo, Passaporte } from "./tipos"

export const CHAVE: string
export function inspecionar(bytes: Uint8Array | ArrayBuffer): InfoModelo
export function carimbar(bytes: Uint8Array | ArrayBuffer, passaporte: Passaporte): Uint8Array
export function problemas(p: unknown, grafo?: Pick<InfoModelo, "entradas" | "saidas"> | null): string[]
