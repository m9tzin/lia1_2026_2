import { useCallback, useEffect, useState } from "react"

import { Carimbo } from "@/components/Carimbo"
import { Ficha } from "@/components/Ficha"
import { Hero } from "@/components/Hero"
import { Resultado } from "@/components/Resultado"
import { Soltar } from "@/components/Soltar"
import { BotaoTema } from "@/components/Tema"
import {
  baixar,
  carimbar,
  criarSessao,
  executar,
  inspecionar,
  problemas,
  rascunho,
  type Execucao,
  type Sessao,
} from "@/lib/motor"
import type { InfoModelo, Passaporte } from "@/lib/tipos"
import { cn } from "@/lib/utils"

type Aviso = { texto: string; tipo: "erro" | "info" } | null

function Controle(props: {
  rotulo: string
  valor: number
  min: number
  max: number
  passo: number
  formatar: (v: number) => string
  aoMudar: (v: number) => void
}) {
  return (
    <label className="block border-b px-5 py-4 last:border-b-0">
      <div className="flex items-baseline justify-between">
        <span className="rotulo">{props.rotulo}</span>
        <span className="font-mono text-sm tabular-nums">{props.formatar(props.valor)}</span>
      </div>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.passo}
        value={props.valor}
        onChange={(e) => props.aoMudar(Number(e.target.value))}
        className="mt-3 w-full accent-foreground"
      />
    </label>
  )
}

export default function App() {
  const [bytes, setBytes] = useState<Uint8Array | null>(null)
  const [nomeModelo, setNomeModelo] = useState<string>()
  const [info, setInfo] = useState<InfoModelo | null>(null)
  const [passaporte, setPassaporte] = useState<Passaporte | null>(null)
  const [pendencia, setPendencia] = useState<{ inicial: Passaporte; motivo: string | null } | null>(null)
  const [sessao, setSessao] = useState<Sessao | null>(null)
  const [imagem, setImagem] = useState<HTMLImageElement | null>(null)
  const [nomeImagem, setNomeImagem] = useState<string>()
  const [execucao, setExecucao] = useState<Execucao | null>(null)
  const [rodada, setRodada] = useState(0)
  const [rodando, setRodando] = useState(false)
  const [confianca, setConfianca] = useState(0.25)
  const [iou, setIou] = useState(0.45)
  const [aviso, setAviso] = useState<Aviso>(null)

  const ativar = useCallback(async (b: Uint8Array, p: Passaporte) => {
    setPassaporte(p)
    setPendencia(null)
    setConfianca(p.operacao.confianca_minima)
    setIou(p.operacao.iou_nms ?? 0.45)
    setAviso({ texto: "Criando sessão do ONNX Runtime…", tipo: "info" })
    try {
      setSessao(await criarSessao(b))
      setAviso(null)
    } catch (e) {
      setAviso({ texto: `ONNX Runtime recusou o modelo: ${(e as Error).message}`, tipo: "erro" })
    }
  }, [])

  async function carregarModelo(arquivo: File) {
    const b = new Uint8Array(await arquivo.arrayBuffer())
    setBytes(b)
    setNomeModelo(arquivo.name)
    setSessao(null)
    setExecucao(null)
    setPassaporte(null)
    let i: InfoModelo
    try {
      i = inspecionar(b)
    } catch (e) {
      setInfo(null)
      setAviso({ texto: `Não é um ONNX válido: ${(e as Error).message}`, tipo: "erro" })
      return
    }
    setInfo(i)
    if (!i.passaporte) {
      setPendencia({ inicial: rascunho(i, arquivo.name), motivo: i.erroPassaporte })
      setAviso(null)
      return
    }
    const erros = problemas(i.passaporte, i)
    if (erros.length) {
      setPendencia({ inicial: i.passaporte, motivo: `Passaporte com problemas: ${erros.join("; ")}` })
      return
    }
    await ativar(b, i.passaporte)
  }

  async function carimbarEAtivar(p: Passaporte) {
    if (!bytes || !nomeModelo) return
    const novo = carimbar(bytes, p)
    setBytes(novo)
    setInfo(inspecionar(novo))
    baixar(novo, nomeModelo.replace(/\.onnx$/, "") + "_carimbado.onnx")
    await ativar(novo, p)
  }

  async function carregarImagem(arquivo: File) {
    const img = new Image()
    img.src = URL.createObjectURL(arquivo)
    await img.decode()
    setNomeImagem(arquivo.name)
    setImagem(img)
  }

  // Roda a inferência sempre que houver sessão e imagem novas. Limiar e IoU só re-decodificam.
  useEffect(() => {
    if (!sessao || !passaporte || !imagem) return
    let vivo = true
    setRodando(true)
    executar(sessao, passaporte, imagem)
      .then((ex) => {
        if (!vivo) return
        setExecucao(ex)
        setRodada((n) => n + 1)
      })
      .catch((e) => vivo && setAviso({ texto: `Falha na inferência: ${(e as Error).message}`, tipo: "erro" }))
      .finally(() => vivo && setRodando(false))
    return () => {
      vivo = false
    }
  }, [sessao, passaporte, imagem])

  const mb = bytes ? `${(bytes.length / 1e6).toFixed(1)} MB` : undefined

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-[1240px] border-x">
        <header className="flex items-center justify-between border-b px-6 py-4 sm:px-10">
          <div className="flex items-center gap-2.5">
            <span className="grid size-6 place-items-center bg-neon font-mono text-[11px] font-bold text-black">P</span>
            <span className="text-sm font-medium tracking-tight">passaporte/onnx</span>
          </div>
          <nav className="flex items-center gap-5">
            <span className="rotulo hidden sm:inline">LIA 1 · 2026/2</span>
            <a
              href="https://github.com/m9tzin/lia1_2026_2"
              target="_blank"
              rel="noreferrer"
              className="rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors hover:bg-foreground hover:text-background"
            >
              GitHub
            </a>
            <BotaoTema />
          </nav>
        </header>

        <Hero />

        <div className="grid border-b sm:grid-cols-2">
          <div className="border-b sm:border-r sm:border-b-0">
            <Soltar
              id="arquivo-modelo"
              numero="01"
              titulo="Modelo"
              aceita=".onnx"
              arquivo={nomeModelo}
              detalhe={mb && `${mb} · ${info?.produtor || "produtor desconhecido"}`}
              aoReceber={carregarModelo}
            />
          </div>
          <Soltar
            id="arquivo-imagem"
            numero="02"
            titulo="Imagem"
            aceita="image/*"
            arquivo={nomeImagem}
            detalhe={imagem && `${imagem.naturalWidth}×${imagem.naturalHeight} px`}
            aoReceber={carregarImagem}
          />
        </div>

        {aviso && (
          <div className={cn("border-b px-6 py-3 font-mono text-xs sm:px-10", aviso.tipo === "erro" ? "text-alerta" : "text-muted-foreground")}>
            {aviso.tipo === "erro" ? "× " : "› "}
            {aviso.texto}
          </div>
        )}

        <main className="grid lg:grid-cols-[400px_1fr]">
          <aside className="border-b lg:border-r lg:border-b-0">
            {pendencia && info ? (
              <Carimbo key={nomeModelo} info={info} inicial={pendencia.inicial} motivo={pendencia.motivo} aoCarimbar={carimbarEAtivar} />
            ) : passaporte ? (
              <>
                <Ficha p={passaporte} />
                <div className="border-t">
                  <div className="border-b px-5 py-3">
                    <span className="rotulo">Operação</span>
                  </div>
                  <Controle
                    rotulo="Confiança mínima"
                    valor={confianca}
                    min={0.05}
                    max={0.95}
                    passo={0.05}
                    formatar={(v) => `${Math.round(v * 100)}%`}
                    aoMudar={setConfianca}
                  />
                  {passaporte.tarefa === "deteccao" && (
                    <Controle rotulo="IoU do NMS" valor={iou} min={0.1} max={0.9} passo={0.05} formatar={(v) => v.toFixed(2)} aoMudar={setIou} />
                  )}
                </div>
              </>
            ) : (
              <div className="px-5 py-10">
                <span className="rotulo">Passaporte</span>
                <p className="mt-3 text-sm text-muted-foreground">
                  Solte um <code className="font-mono text-foreground">.onnx</code> para ler o contrato gravado em{" "}
                  <code className="font-mono text-foreground">lia.passaporte</code>.
                </p>
              </div>
            )}
          </aside>
          <section className="min-w-0">
            <Resultado
              imagem={imagem}
              passaporte={passaporte}
              execucao={execucao}
              rodada={rodada}
              rodando={rodando}
              confianca={confianca}
              iou={iou}
            />
          </section>
        </main>

        <footer className="flex flex-wrap items-center justify-between gap-2 border-t px-6 py-5 sm:px-10">
          <span className="rotulo">Matheus Sousa Marinho · entrega 03</span>
          <span className="tag text-muted-foreground">onnx runtime web · inferência local, nada sai do navegador</span>
        </footer>
      </div>
    </div>
  )
}
