import { type ComponentPropsWithoutRef, type FC, useRef, useState } from "react"
import { motion, useInView, useReducedMotion } from "motion/react"

import { cn } from "@/lib/utils"

// Laço feito "à mão": a ponta final passa do início, como uma caneta que não fecha certinho.
const LACO =
  "M150 8C185 10 198 30 194 46C188 68 140 76 96 75C50 74 8 66 5 44C2 22 40 6 100 5C130 4 160 8 178 18"

export interface CircledTextProps extends ComponentPropsWithoutRef<"span"> {
  delay?: number
  duration?: number
  strokeWidth?: number
}

export const CircledText: FC<CircledTextProps> = ({
  children,
  className,
  delay = 0.6,
  duration = 1.1,
  strokeWidth = 3,
  ...props
}) => {
  const reduzir = useReducedMotion()
  // O laço se apaga quando a palavra sai da tela e volta a ser desenhado quando ela reaparece.
  const ref = useRef<HTMLSpanElement>(null)
  const visivel = useInView(ref)
  // Só o primeiro desenho espera o título aparecer; os seguintes respondem à rolagem na hora.
  const [primeiro, setPrimeiro] = useState(true)

  return (
    <span ref={ref} className={cn("relative isolate inline-block whitespace-nowrap", className)} {...props}>
      {children}
      <svg
        aria-hidden
        viewBox="0 0 200 80"
        preserveAspectRatio="none"
        className="pointer-events-none absolute -inset-x-[0.3em] -inset-y-[0.18em] -z-10 h-[calc(100%+0.36em)] w-[calc(100%+0.6em)] overflow-visible text-marca"
      >
        <motion.path
          d={LACO}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
          initial={{ pathLength: reduzir ? 1 : 0 }}
          animate={{ pathLength: visivel ? 1 : 0 }}
          transition={
            reduzir
              ? { duration: 0 }
              : { delay: primeiro ? delay : 0, duration: primeiro ? duration : 0.5, ease: [0.65, 0, 0.35, 1] }
          }
          onAnimationComplete={() => visivel && setPrimeiro(false)}
        />
      </svg>
    </span>
  )
}
