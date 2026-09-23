import { AnimatedShinyText } from "@/components/ui/animated-shiny-text"
import { FundoAscii } from "@/components/FundoAscii"
import { CircledText } from "@/components/ui/circled-text"
import { TypingAnimation } from "@/components/ui/typing-animation"

// O mesmo fundo ASCII das laterais, aqui no amarelo neon dos antigos blocos da grade. No tema claro
// o neon some no branco, então os caracteres usam um oliva mais escuro.
const NEON = { dark: "#f7fe62", light: "#9fa300" }

// Só a metade direita recebe o ASCII, para não competir com o título. A máscara horizontal some
// da direita para o meio e a vertical esmaece em direção ao rodapé do hero; as duas se intersectam.
const MASCARA = {
  maskImage: "linear-gradient(to left, black 25%, transparent 60%), linear-gradient(to bottom, black 40%, transparent)",
  maskComposite: "intersect",
} as const

export function Hero() {
  return (
    <section className="relative overflow-hidden border-b">
      <div className="absolute inset-0" style={MASCARA}>
        <FundoAscii cores={NEON} className="absolute inset-0" />
      </div>
      {/* O bloco deixa o cursor passar até o ASCII; o texto em si continua selecionável. */}
      <div className="pointer-events-none relative *:pointer-events-auto px-6 pt-20 pb-14 sm:px-10 sm:pt-24 sm:pb-20">
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
