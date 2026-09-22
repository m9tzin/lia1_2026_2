// Pré e pós-processamento dirigidos pelo passaporte. Nada aqui sabe de YOLO, Keras ou PyTorch:
// só de "entrada.redimensionar" e "saida.formato". Funções puras, sem DOM, testadas em Node.

// Onde a imagem original (w x h) cai dentro do tensor de entrada (W x H).
// Devolve o retângulo de desenho {dx, dy, dw, dh} e o fator para desfazer a transformação.
export function geometria(w, h, entrada) {
  const [H, W] = entrada.tamanho;
  if (entrada.redimensionar === "esticar") {
    return { W, H, dx: 0, dy: 0, dw: W, dh: H, sx: W / w, sy: H / h };
  }
  if (entrada.redimensionar === "recorte_central") {
    const r = Math.max(W / w, H / h);
    const dw = Math.round(w * r);
    const dh = Math.round(h * r);
    return { W, H, dx: (W - dw) / 2, dy: (H - dh) / 2, dw, dh, sx: r, sy: r };
  }
  // letterbox: mantém a proporção e preenche as bordas, como o Ultralytics
  const r = Math.min(W / w, H / h);
  const dw = Math.round(w * r);
  const dh = Math.round(h * r);
  const dx = Math.max(0, Math.round((W - dw) / 2 - 0.1));
  const dy = Math.max(0, Math.round((H - dh) / 2 - 0.1));
  return { W, H, dx, dy, dw, dh, sx: dw / w, sy: dh / h };
}

// Pixels RGBA (já redimensionados para W x H) -> Float32Array no layout e normalização declarados.
export function paraTensor(rgba, entrada) {
  const [H, W] = entrada.tamanho;
  const n = W * H;
  const saida = new Float32Array(3 * n);
  const ordem = entrada.cores === "BGR" ? [2, 1, 0] : [0, 1, 2];
  const { escala, media, desvio } = entrada;
  const nchw = entrada.layout === "NCHW";
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = (rgba[i * 4 + ordem[c]] * escala - media[c]) / desvio[c];
      saida[nchw ? c * n + i : i * 3 + c] = v;
    }
  }
  return { dados: saida, dims: nchw ? [1, 3, H, W] : [1, H, W, 3] };
}

// Caixa no espaço do tensor -> caixa na imagem original, limitada às bordas.
export function desfazer([x1, y1, x2, y2], geo, w, h) {
  const lim = (v, max) => Math.min(Math.max(v, 0), max);
  return [
    lim((x1 - geo.dx) / geo.sx, w),
    lim((y1 - geo.dy) / geo.sy, h),
    lim((x2 - geo.dx) / geo.sx, w),
    lim((y2 - geo.dy) / geo.sy, h),
  ];
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return uniao > 0 ? inter / uniao : 0;
}

// Supressão de não máximos por classe.
export function nms(deteccoes, limiarIou) {
  const ordenadas = [...deteccoes].sort((a, b) => b.confianca - a.confianca);
  const mantidas = [];
  for (const d of ordenadas) {
    if (mantidas.every((m) => m.indice !== d.indice || iou(m.caixa, d.caixa) < limiarIou)) {
      mantidas.push(d);
    }
  }
  return mantidas;
}

// [1, 4+nc, N] (ou transposto [1, N, 4+nc]) com cx, cy, w, h e uma pontuação por classe.
function caixasXywhPorClasse(dados, dims, nc, piso) {
  const canais = 4 + nc;
  let N;
  let em;
  if (dims[1] === canais) {
    N = dims[2];
    em = (canal, i) => dados[canal * N + i];
  } else if (dims[2] === canais) {
    N = dims[1];
    em = (canal, i) => dados[i * canais + canal];
  } else {
    throw new Error(`saída ${JSON.stringify(dims)} não tem 4 + ${nc} canais`);
  }
  const brutas = [];
  for (let i = 0; i < N; i++) {
    let melhor = 0;
    let indice = 0;
    for (let c = 0; c < nc; c++) {
      const s = em(4 + c, i);
      if (s > melhor) {
        melhor = s;
        indice = c;
      }
    }
    if (melhor < piso) continue;
    const cx = em(0, i);
    const cy = em(1, i);
    const bw = em(2, i);
    const bh = em(3, i);
    brutas.push({ indice, confianca: melhor, caixa: [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2] });
  }
  return brutas;
}

// [1, K, 6] com x1, y1, x2, y2, confiança, classe: saída já filtrada (YOLO26, YOLOv10).
function caixasXyxyConfClasse(dados, dims, piso) {
  const K = dims[1];
  const brutas = [];
  for (let i = 0; i < K; i++) {
    const o = i * 6;
    if (dados[o + 4] < piso) continue;
    brutas.push({
      indice: Math.round(dados[o + 5]),
      confianca: dados[o + 4],
      caixa: [dados[o], dados[o + 1], dados[o + 2], dados[o + 3]],
    });
  }
  return brutas;
}

export function softmax(valores) {
  const max = Math.max(...valores);
  const exps = valores.map((v) => Math.exp(v - max));
  const soma = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / soma);
}

// Pisos de confiança: o leitor decodifica abaixo do limiar de exibição para que a decisão
// consiga enxergar candidatas na zona incerta.
export function pisoDeDecodificacao(passaporte, confianca) {
  const d = passaporte.decisao;
  const pisos = [confianca];
  if (d) pisos.push(d.limiar_alerta, ...(d.zona_incerta ?? []));
  return Math.min(...pisos);
}

// Saída bruta -> resultado interpretado.
// deteccao: {tipo, deteccoes (>= confianca), candidatas (>= piso)}; classificacao: {tipo, ranking}.
export function decodificar(saida, passaporte, geo, w, h, { confianca, iou: limiarIou }) {
  const { formato } = passaporte.saida;
  const classes = passaporte.classes;
  const nome = (i) => classes[i] ?? `classe_${i}`;

  if (formato === "probabilidades" || formato === "logits") {
    const valores = Array.from(saida.dados);
    const probs = formato === "logits" ? softmax(valores) : valores;
    const ranking = probs
      .map((p, i) => ({ indice: i, classe: nome(i), confianca: p }))
      .sort((a, b) => b.confianca - a.confianca);
    return { tipo: "classificacao", ranking };
  }

  const piso = pisoDeDecodificacao(passaporte, confianca);
  let brutas;
  if (formato === "caixas_xywh_por_classe") {
    brutas = nms(caixasXywhPorClasse(saida.dados, saida.dims, classes.length, piso), limiarIou);
  } else if (formato === "caixas_xyxy_conf_classe") {
    brutas = caixasXyxyConfClasse(saida.dados, saida.dims, piso);
  } else {
    throw new Error(`formato de saída desconhecido: ${formato}`);
  }
  const candidatas = brutas
    .map((d) => ({ ...d, classe: nome(d.indice), caixa: desfazer(d.caixa, geo, w, h) }))
    .sort((a, b) => b.confianca - a.confianca);
  return {
    tipo: "deteccao",
    deteccoes: candidatas.filter((d) => d.confianca >= confianca),
    candidatas,
  };
}

// Aplica a regra de produto do passaporte.
// Estados: ALERTA, VERIFICAR, OK (com bloco decisao) e INCERTO (classificação abaixo do limiar).
export function decidir(resultado, passaporte, confianca) {
  const d = passaporte.decisao;
  const rotuloIncerto = passaporte.operacao?.rotulo_incerto ?? "INCERTO";

  if (resultado.tipo === "classificacao") {
    const top = resultado.ranking[0];
    if (top.confianca < confianca) {
      return { estado: rotuloIncerto, motivo: `top 1 (${top.classe}) abaixo de ${pct(confianca)}` };
    }
    if (!d) return { estado: top.classe, motivo: `${pct(top.confianca)} de confiança` };
    if (d.alertar_se.includes(top.classe) && top.confianca >= d.limiar_alerta) {
      return { estado: "ALERTA", motivo: `${d.mensagem}: ${top.classe} ${pct(top.confianca)}` };
    }
    return { estado: "OK", motivo: `${top.classe} ${pct(top.confianca)}` };
  }

  if (!d) {
    const n = resultado.deteccoes.length;
    return { estado: n ? `${n} objeto(s)` : "NADA", motivo: n ? "" : `nada acima de ${pct(confianca)}` };
  }
  const alvo = resultado.candidatas.filter((c) => d.alertar_se.includes(c.classe));
  const melhor = alvo[0];
  if (melhor && melhor.confianca >= d.limiar_alerta) {
    const n = alvo.filter((c) => c.confianca >= d.limiar_alerta).length;
    return { estado: "ALERTA", motivo: `${d.mensagem} (${n}, maior ${pct(melhor.confianca)})`, alvo: melhor };
  }
  const zona = d.zona_incerta;
  if (melhor && zona && melhor.confianca >= zona[0] && melhor.confianca <= zona[1]) {
    return { estado: "VERIFICAR", motivo: `${melhor.classe} com ${pct(melhor.confianca)}, dentro da zona incerta`, alvo: melhor };
  }
  return { estado: "OK", motivo: `nenhum ${d.alertar_se.join(" / ")} acima de ${pct(zona?.[0] ?? d.limiar_alerta)}` };
}

function pct(v) {
  return `${Math.round(v * 100)}%`;
}
