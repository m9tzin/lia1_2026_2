import { Moon, Sun } from "lucide-react"
import { useEffect, useState } from "react"

type Tema = "light" | "dark"

function inicial(): Tema {
  try {
    const salvo = localStorage.getItem("tema")
    if (salvo === "light" || salvo === "dark") return salvo
  } catch {
    // armazenamento bloqueado: segue o sistema
  }
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
}

// Alterna claro/escuro; começa pelo tema do sistema e lembra a escolha neste navegador.
export function BotaoTema() {
  const [tema, setTema] = useState<Tema>(inicial)

  useEffect(() => {
    document.documentElement.dataset.theme = tema
    try {
      localStorage.setItem("tema", tema)
    } catch {
      // sem persistência, sem problema
    }
  }, [tema])

  const Icone = tema === "dark" ? Sun : Moon
  return (
    <button
      type="button"
      onClick={() => setTema(tema === "dark" ? "light" : "dark")}
      aria-label={tema === "dark" ? "Usar tema claro" : "Usar tema escuro"}
      className="grid size-8 place-items-center rounded-full border transition-colors hover:bg-foreground hover:text-background"
    >
      <Icone className="size-3.5" />
    </button>
  )
}
