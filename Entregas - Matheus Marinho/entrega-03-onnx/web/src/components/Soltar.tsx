import { useRef, useState, type ReactNode } from "react"

import { DotPattern } from "@/components/ui/dot-pattern"
import { cn } from "@/lib/utils"

interface Props {
  numero: string
  titulo: string
  aceita: string
  arquivo?: string
  detalhe?: ReactNode
  aoReceber: (arquivo: File) => void
  id: string
}

// Área de soltar arquivo. O padrão de pontos aparece ao arrastar ou passar o mouse.
export function Soltar({ numero, titulo, aceita, arquivo, detalhe, aoReceber, id }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [arrastando, setArrastando] = useState(false)

  return (
    <label
      htmlFor={id}
      onDragOver={(e) => {
        e.preventDefault()
        setArrastando(true)
      }}
      onDragLeave={() => setArrastando(false)}
      onDrop={(e) => {
        e.preventDefault()
        setArrastando(false)
        const f = e.dataTransfer.files[0]
        if (f) aoReceber(f)
      }}
      className={cn(
        "group relative flex min-h-36 cursor-pointer flex-col justify-between overflow-hidden p-6 transition-colors",
        arrastando ? "bg-neon/25" : "hover:bg-muted",
      )}
    >
      <DotPattern
        width={14}
        height={14}
        cr={1}
        className={cn(
          "text-foreground/15 transition-opacity duration-300",
          arrastando ? "opacity-100" : "opacity-0 group-hover:opacity-100",
        )}
      />
      <input
        ref={input}
        id={id}
        type="file"
        accept={aceita}
        className="sr-only"
        onChange={() => {
          const f = input.current?.files?.[0]
          if (f) aoReceber(f)
          // Zera o campo para o mesmo arquivo poder ser escolhido de novo depois de limpar.
          if (input.current) input.current.value = ""
        }}
      />
      <div className="relative flex items-center justify-between">
        <span className="rotulo">
          {numero} / {titulo}
        </span>
        {arquivo ? <span className="tag text-ok">[carregado]</span> : <span className="tag text-muted-foreground">[vazio]</span>}
      </div>
      <div className="relative mt-6">
        <p className="truncate text-lg font-medium tracking-tight">
          {arquivo ?? (
            <>
              Solte aqui ou <span className="underline decoration-neon decoration-4 underline-offset-4">escolha</span>
            </>
          )}
        </p>
        {detalhe && <div className="mt-1 text-sm text-muted-foreground">{detalhe}</div>}
      </div>
    </label>
  )
}
