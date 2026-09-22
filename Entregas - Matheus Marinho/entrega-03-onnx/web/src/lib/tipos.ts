// Tipos do passaporte v1 (espelham passaporte/esquema_v1.json) e do resultado da inferência.

export type Formato = "caixas_xywh_por_classe" | "caixas_xyxy_conf_classe" | "probabilidades" | "logits"

export interface Passaporte {
  versao: 1
  nome: string
  descricao?: string
  tarefa: "deteccao" | "classificacao"
  classes: string[]
  entrada: {
    tensor: string
    layout: "NCHW" | "NHWC"
    tamanho: [number, number]
    cores: "RGB" | "BGR"
    redimensionar: "letterbox" | "esticar" | "recorte_central"
    preenchimento?: number
    escala: number
    media: [number, number, number]
    desvio: [number, number, number]
  }
  saida: { tensor: string; formato: Formato }
  operacao: { confianca_minima: number; iou_nms?: number; rotulo_incerto?: string }
  decisao?: {
    alertar_se: string[]
    limiar_alerta: number
    zona_incerta?: [number, number]
    mensagem: string
  }
  origem?: {
    framework?: string
    pesos?: string
    dataset?: string
    exportado_em?: string
    best?: {
      epoca?: number
      epocas_treinadas?: number
      mAP50?: number
      mAP50_95?: number
      precisao?: number
      recall?: number
      fitness?: number
    }
    [chave: string]: unknown
  }
}

export interface TensorInfo {
  nome: string
  dims: (number | string | null)[]
}

export interface InfoModelo {
  produtor: string
  metadados: Record<string, string>
  passaporte: Passaporte | null
  erroPassaporte: string | null
  entradas: TensorInfo[]
  saidas: TensorInfo[]
}

export interface Deteccao {
  indice: number
  classe: string
  confianca: number
  caixa: [number, number, number, number]
}

export type Resultado =
  | { tipo: "deteccao"; deteccoes: Deteccao[]; candidatas: Deteccao[] }
  | { tipo: "classificacao"; ranking: { indice: number; classe: string; confianca: number }[] }

export interface Decisao {
  estado: string
  motivo: string
  alvo?: Deteccao
}

export interface Geometria {
  W: number
  H: number
  dx: number
  dy: number
  dw: number
  dh: number
  sx: number
  sy: number
}

export interface SaidaBruta {
  dados: Float32Array
  dims: readonly number[]
}
