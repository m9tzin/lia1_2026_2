import { useEffect, useRef } from "react"

import { mountFluidBackground } from "@/components/ascii/fluid-background"
import { useTemaAtual } from "@/components/Tema"
import { cn } from "@/lib/utils"

type Cores = { dark: string; light: string }

// Uma cor só para todos os caracteres, bem perto do fundo: um pouco mais clara que o preto no
// tema escuro e um pouco mais escura que o branco no claro.
const COR_GLIFO: Cores = { dark: "#1e1e1e", light: "#e9e9e7" }

// Por padrão, camada fixa atrás da página inteira. A coluna central tem fundo sólido e cobre a
// animação, então o ASCII só aparece nas faixas laterais. Essas faixas deixam o cursor passar até aqui.
// Com `className`, a mesma animação preenche um bloco da página, como o hero.
export function FundoAscii({ cores = COR_GLIFO, className = "fixed inset-0 z-0" }: { cores?: Cores; className?: string }) {
  const host = useRef<HTMLDivElement>(null)
  const tela = useRef<HTMLCanvasElement>(null)
  const tema = useTemaAtual()

  useEffect(() => {
    if (!host.current || !tela.current) return
    const fundo = mountFluidBackground(host.current, tela.current, {
      colorMode: "accent",
      accentColor: cores[tema],
      fps: 30,
    })
    return () => fundo?.destroy()
  }, [tema, cores])

  return (
    <div ref={host} aria-hidden="true" className={cn(className)}>
      <canvas ref={tela} className="pointer-events-none absolute inset-0 size-full" />
    </div>
  )
}
