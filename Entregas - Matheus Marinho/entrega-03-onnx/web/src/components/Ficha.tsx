import type { ReactNode } from "react"

import { NumberTicker } from "@/components/ui/number-ticker"
import type { Passaporte } from "@/lib/tipos"

function Linha({ rotulo, children }: { rotulo: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[96px_1fr] gap-3 border-b px-5 py-3 last:border-b-0">
      <dt className="rotulo pt-0.5">{rotulo}</dt>
      <dd className="min-w-0 text-sm break-words">{children}</dd>
    </div>
  )
}

function Metrica({ rotulo, valor, casas = 3 }: { rotulo: string; valor?: number; casas?: number }) {
  return (
    <div className="border-r px-5 py-4 last:border-r-0">
      <div className="rotulo">{rotulo}</div>
      <div className="mt-1 font-mono text-2xl font-medium tracking-tight tabular-nums">
        {valor == null ? <span className="text-muted-foreground">n/d</span> : <NumberTicker value={valor} decimalPlaces={casas} />}
      </div>
    </div>
  )
}

// Ficha técnica do passaporte: o que o leitor vai seguir.
export function Ficha({ p }: { p: Passaporte }) {
  const e = p.entrada
  const o = p.origem ?? {}
  const b = o.best
  const d = p.decisao
  const visiveis = p.classes.slice(0, 24)

  return (
    <div>
      <div className="flex items-center justify-between border-b px-5 py-3">
        <span className="rotulo">Passaporte</span>
        <span className="tag bg-neon px-1.5 py-0.5 text-black">{p.tarefa}</span>
      </div>
      <div className="px-5 pt-5 pb-4">
        <h2 className="text-xl font-medium tracking-tight">{p.nome}</h2>
        {o.framework && <p className="mt-1 font-mono text-xs text-muted-foreground">{o.framework}</p>}
      </div>

      {b && (
        <div className="grid grid-cols-3 border-y">
          <Metrica rotulo="mAP50" valor={b.mAP50} />
          <Metrica rotulo="mAP50-95" valor={b.mAP50_95} />
          <Metrica rotulo="Época" valor={b.epoca} casas={0} />
        </div>
      )}

      <dl className={b ? "" : "border-t"}>
        <Linha rotulo={`Classes ${p.classes.length}`}>
          <div className="flex flex-wrap gap-1">
            {visiveis.map((c) => (
              <span key={c} className="border px-1.5 py-px font-mono text-[11px]">
                {c}
              </span>
            ))}
            {p.classes.length > visiveis.length && (
              <span className="px-1 font-mono text-[11px] text-muted-foreground">
                +{p.classes.length - visiveis.length}
              </span>
            )}
          </div>
        </Linha>
        <Linha rotulo="Entrada">
          <span className="font-mono text-xs">
            {e.tensor} · {e.layout} · {e.tamanho.join("×")} · {e.cores} · {e.redimensionar}
          </span>
        </Linha>
        <Linha rotulo="Saída">
          <span className="font-mono text-xs">
            {p.saida.tensor} · {p.saida.formato}
          </span>
        </Linha>
        {d && (
          <Linha rotulo="Decisão">
            <span className="text-alerta">{d.mensagem}</span>
            <span className="text-muted-foreground">
              {" "}
              quando {d.alertar_se.join(", ")} ≥ {d.limiar_alerta}
              {d.zona_incerta && `; verificar entre ${d.zona_incerta[0]} e ${d.zona_incerta[1]}`}
            </span>
          </Linha>
        )}
        {(o.dataset || o.pesos) && (
          <Linha rotulo="Origem">
            <span className="font-mono text-xs">{[o.pesos, o.dataset].filter(Boolean).join(" · ")}</span>
          </Linha>
        )}
        {o.exportado_em && (
          <Linha rotulo="Exportado">
            <span className="font-mono text-xs">{o.exportado_em}</span>
          </Linha>
        )}
      </dl>
    </div>
  )
}
