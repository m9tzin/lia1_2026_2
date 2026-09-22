import { AnimatedShinyText } from "@/components/ui/animated-shiny-text"
import { CircledText } from "@/components/ui/circled-text"
import { GridPattern } from "@/components/ui/grid-pattern"
import { TypingAnimation } from "@/components/ui/typing-animation"

// Quadrados acesos na grade, como os blocos neon do Infisical.
const QUADRADOS: [number, number][] = [
  [9, 0], [10, 0], [16, 0], [24, 1], [25, 1], [25, 2], [22, 3], [27, 4], [23, 6], [26, 7],
]

export function Hero() {
  return (
    <section className="relative overflow-hidden border-b">
      <GridPattern
        width={44}
        height={44}
        squares={QUADRADOS}
        className="fill-neon stroke-(--grade) mask-[linear-gradient(to_bottom,black_40%,transparent)]"
      />
      <div className="relative px-6 pt-20 pb-14 sm:px-10 sm:pt-24 sm:pb-20">
        <div className="inline-flex items-center gap-2 rounded-full border bg-background px-3 py-1">
          <span className="size-1.5 rounded-full bg-ok" />
          <AnimatedShinyText className="tag mx-0 max-w-none">YOLONNX v1</AnimatedShinyText>
        </div>

        <h1 className="mt-6 max-w-3xl text-4xl leading-[1.05] font-medium tracking-[-0.03em] sm:text-6xl">
          leitor universal de modelos <CircledText>.onnx</CircledText>
        </h1>

        <p className="mt-5 max-w-2xl text-base text-muted-foreground sm:text-lg">
          O próprio arquivo diz como deve ser usado:{" "}
          <TypingAnimation
            as="span"
            className="font-mono text-[0.95em] font-medium text-foreground"
            words={["as classes.", "o pré-processamento.", "a regra de decisão.", "as métricas do best.pt."]}
            typeSpeed={45}
            deleteSpeed={25}
            pauseDelay={1600}
            loop
            cursorStyle="block"
          />
        </p>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          Sem <code className="font-mono text-foreground">classes.txt</code> ao lado, sem código específico de
          framework. Tudo roda no navegador.
        </p>
      </div>
    </section>
  )
}
