import { useState } from "react"

import { ShimmerButton } from "@/components/ui/shimmer-button"
import type { InfoModelo, Passaporte } from "@/lib/tipos"
import { problemas } from "@/lib/motor"

interface Props {
  info: InfoModelo
  inicial: Passaporte
  motivo?: string | null
  aoCarimbar: (p: Passaporte) => void
}

// Fallback de contrato: o modelo não diz como deve ser usado, então o usuário completa o passaporte.
export function Carimbo({ info, inicial, motivo, aoCarimbar }: Props) {
  const [texto, setTexto] = useState(() => JSON.stringify(inicial, null, 2))
  const [erros, setErros] = useState<string[]>([])

  function carimbar() {
    let p: Passaporte
    try {
      p = JSON.parse(texto)
    } catch (e) {
      setErros([`JSON inválido: ${(e as Error).message}`])
      return
    }
    const lista = problemas(p, info)
    setErros(lista)
    if (!lista.length) aoCarimbar(p)
  }

  return (
    <div>
      <div className="flex items-center justify-between border-b px-5 py-3">
        <span className="rotulo">Sem passaporte</span>
        <span className="tag text-verificar">[aguardando contrato]</span>
      </div>
      <div className="space-y-3 px-5 py-5 text-sm">
        <p>
          Este modelo não diz como deve ser usado, e o leitor não adivinha. O rascunho abaixo já traz os
          tensores do grafo; confira as classes e o pré-processamento antes de carimbar.
        </p>
        {motivo && <p className="text-verificar">{motivo}</p>}
        <div className="flex flex-wrap gap-1.5">
          {[...info.entradas.map((t) => ["entrada", t] as const), ...info.saidas.map((t) => ["saída", t] as const)].map(
            ([tipo, t]) => (
              <span key={tipo + t.nome} className="border px-1.5 py-0.5 font-mono text-[11px]">
                {tipo} {t.nome} [{t.dims.join(", ")}]
              </span>
            ),
          )}
        </div>
      </div>
      <textarea
        value={texto}
        onChange={(e) => setTexto(e.target.value)}
        spellCheck={false}
        className="block h-[360px] w-full resize-y border-y bg-muted px-5 py-4 font-mono text-xs leading-relaxed outline-none focus:bg-background"
      />
      {erros.length > 0 && (
        <ul className="space-y-1 border-b px-5 py-3 font-mono text-xs text-alerta">
          {erros.map((e) => (
            <li key={e}>× {e}</li>
          ))}
        </ul>
      )}
      <div className="px-5 py-5">
        <ShimmerButton
          onClick={carimbar}
          background="var(--foreground)"
          shimmerColor="#f7fe62"
          shimmerDuration="2.5s"
          className="w-full text-sm font-medium text-background"
        >
          Carimbar e baixar
        </ShimmerButton>
      </div>
    </div>
  )
}
