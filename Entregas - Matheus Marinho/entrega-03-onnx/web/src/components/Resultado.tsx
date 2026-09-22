import { useEffect, useMemo, useRef } from "react"

import { AnimatedList } from "@/components/ui/animated-list"
import { BorderBeam } from "@/components/ui/border-beam"
import { NumberTicker } from "@/components/ui/number-ticker"
import { decidir, decodificar } from "@/lib/decodificadores.js"
import type { Execucao } from "@/lib/motor"
import type { Decisao, Deteccao, Passaporte, Resultado as TipoResultado } from "@/lib/tipos"
import { cn } from "@/lib/utils"

interface Props {
  imagem: HTMLImageElement | null
  passaporte: Passaporte | null
  execucao: Execucao | null
  rodada: number
  rodando: boolean
  confianca: number
  iou: number
}

const COR_ESTADO: Record<string, string> = {
  ALERTA: "bg-alerta text-white",
  VERIFICAR: "bg-verificar text-white",
  INCERTO: "bg-verificar text-white",
  OK: "bg-ok text-white",
}

// Vermelho só para o que dispara o alerta: classe de alerta com confiança acima do limiar.
function disparaAlerta(p: Passaporte | null, classe: string, confianca: number) {
  const d = p?.decisao
  return !!d && d.alertar_se.includes(classe) && confianca >= d.limiar_alerta
}

function desenhar(tela: HTMLCanvasElement, img: HTMLImageElement, r: TipoResultado | null, decisao: Decisao | null, p: Passaporte | null) {
  tela.width = img.naturalWidth
  tela.height = img.naturalHeight
  const ctx = tela.getContext("2d")!
  ctx.drawImage(img, 0, 0)
  if (!r || r.tipo !== "deteccao") return

  const traco = Math.max(2, Math.round(Math.max(tela.width, tela.height) / 320))
  const fonte = traco * 7
  ctx.font = `500 ${fonte}px "JetBrains Mono Variable", ui-monospace, monospace`
  ctx.textBaseline = "middle"
  const extra: (Deteccao & { tracejada?: boolean })[] =
    decisao?.estado === "VERIFICAR" && decisao.alvo ? [{ ...decisao.alvo, tracejada: true }] : []

  for (const d of [...r.deteccoes.map((x) => ({ ...x, tracejada: false })), ...extra]) {
    const [x1, y1, x2, y2] = d.caixa
    const perigo = disparaAlerta(p, d.classe, d.confianca)
    const cor = perigo ? "#d92d20" : "#f7fe62"
    ctx.setLineDash(d.tracejada ? [traco * 4, traco * 3] : [])
    ctx.lineWidth = traco + 2
    ctx.strokeStyle = "rgba(0,0,0,0.55)"
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    ctx.lineWidth = traco
    ctx.strokeStyle = cor
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
    ctx.setLineDash([])

    const rotulo = `${d.classe.toUpperCase()} ${Math.round(d.confianca * 100)}%`
    const larg = ctx.measureText(rotulo).width + traco * 5
    const alt = fonte + traco * 3
    const ty = y1 - alt >= 0 ? y1 - alt : y1
    ctx.fillStyle = cor
    ctx.fillRect(x1 - traco / 2, ty, larg, alt)
    ctx.fillStyle = perigo ? "#ffffff" : "#0a0a0a"
    ctx.fillText(rotulo, x1 + traco * 2, ty + alt / 2)
  }
}

export function Resultado({ imagem, passaporte, execucao, rodada, rodando, confianca, iou }: Props) {
  const tela = useRef<HTMLCanvasElement>(null)

  const r = useMemo<TipoResultado | null>(() => {
    if (!execucao || !passaporte || !imagem) return null
    return decodificar(execucao.saida, passaporte, execucao.geo, imagem.naturalWidth, imagem.naturalHeight, { confianca, iou })
  }, [execucao, passaporte, imagem, confianca, iou])

  const decisao = useMemo<Decisao | null>(
    () => (r && passaporte ? decidir(r, passaporte, confianca) : null),
    [r, passaporte, confianca],
  )

  useEffect(() => {
    if (tela.current && imagem) desenhar(tela.current, imagem, r, decisao, passaporte)
  }, [imagem, r, decisao, passaporte])

  const itens =
    r?.tipo === "deteccao"
      ? (r.deteccoes.length ? r.deteccoes : r.candidatas.slice(0, 1)).map((d) => ({
          classe: d.classe,
          confianca: d.confianca,
          extra: d.caixa.map((v) => Math.round(v)).join(", "),
          abaixo: r.deteccoes.length === 0,
        }))
      : r?.tipo === "classificacao"
        ? r.ranking.slice(0, 3).map((d) => ({ classe: d.classe, confianca: d.confianca, extra: "", abaixo: false }))
        : []

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b px-5 py-3">
        <span className="rotulo">Resultado</span>
        {execucao && (
          <span className="tag text-muted-foreground">
            <NumberTicker value={execucao.ms} decimalPlaces={execucao.ms < 10 ? 1 : 0} className="text-foreground" /> ms · wasm
          </span>
        )}
      </div>

      {decisao ? (
        <div className="flex items-stretch border-b">
          <div className={cn("flex items-center px-5 font-mono text-sm font-semibold tracking-wide", COR_ESTADO[decisao.estado] ?? "bg-foreground text-background")}>
            {decisao.estado}
          </div>
          <p className="px-4 py-3 text-sm">{decisao.motivo || " "}</p>
        </div>
      ) : null}

      <div className="relative m-5 flex min-h-72 flex-1 items-center justify-center overflow-hidden border bg-muted">
        {imagem ? (
          <canvas ref={tela} className="block h-auto max-h-[70vh] w-auto max-w-full" />
        ) : (
          <p className="rotulo px-6 text-center">Carregue um modelo e uma imagem</p>
        )}
        {rodando && <BorderBeam size={120} duration={3} borderWidth={2} colorFrom="#f7fe62" colorTo="#0a0a0a" />}
      </div>

      {itens.length > 0 && (
        <div className="border-t">
          <div className="grid grid-cols-[1fr_auto] border-b px-5 py-2">
            <span className="rotulo">Classe</span>
            <span className="rotulo">Confiança</span>
          </div>
          <AnimatedList key={rodada} delay={90} className="items-stretch gap-0">
            {[...itens].reverse().map((d, i) => (
              <div key={`${rodada}-${i}`} className="grid grid-cols-[1fr_auto] items-center gap-4 border-b px-5 py-2.5 last:border-b-0">
                <div className="min-w-0">
                  <span className={cn("text-sm font-medium", disparaAlerta(passaporte, d.classe, d.confianca) && "text-alerta")}>{d.classe}</span>
                  {d.abaixo && <span className="tag ml-2 text-verificar">[abaixo do limiar]</span>}
                  {d.extra && <span className="ml-2 font-mono text-[11px] text-muted-foreground">{d.extra}</span>}
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-1.5 w-24 bg-muted">
                    <div
                      className={cn("h-full", disparaAlerta(passaporte, d.classe, d.confianca) ? "bg-alerta" : "bg-foreground")}
                      style={{ width: `${Math.round(d.confianca * 100)}%` }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono text-sm tabular-nums">{(d.confianca * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </AnimatedList>
        </div>
      )}
    </div>
  )
}
