// Leitura e gravação do passaporte direto nos bytes do .onnx.
//
// Um .onnx é uma mensagem protobuf ModelProto. O ONNX Runtime Web não expõe metadata_props,
// então este módulo lê o protobuf por conta própria. Só percorre os campos de topo e pula o
// grafo pelo comprimento, o que é rápido mesmo em modelos grandes.

export const CHAVE = "lia.passaporte";

const CAMPO_PRODUTOR = 2;
const CAMPO_GRAFO = 7;
const CAMPO_METADADOS = 14;
const GRAFO_ENTRADA = 11;
const GRAFO_SAIDA = 12;

const utf8 = new TextDecoder();
const codificador = new TextEncoder();

function lerVarint(bytes, pos) {
  let valor = 0;
  let fator = 1;
  for (;;) {
    if (pos >= bytes.length) throw new Error("protobuf truncado");
    const b = bytes[pos++];
    valor += (b & 0x7f) * fator;
    if (b < 0x80) return [valor, pos];
    fator *= 128;
  }
}

// Itera sobre os campos de uma mensagem: {campo, tipo, inicio, fim, valor?, dados?}.
function* campos(bytes, inicio = 0, fim = bytes.length) {
  let pos = inicio;
  while (pos < fim) {
    const comeco = pos;
    let tag;
    [tag, pos] = lerVarint(bytes, pos);
    const campo = Math.floor(tag / 8);
    const tipo = tag % 8;
    if (tipo === 0) {
      let valor;
      [valor, pos] = lerVarint(bytes, pos);
      yield { campo, tipo, inicio: comeco, fim: pos, valor };
    } else if (tipo === 2) {
      let tamanho;
      [tamanho, pos] = lerVarint(bytes, pos);
      const dados = bytes.subarray(pos, pos + tamanho);
      pos += tamanho;
      yield { campo, tipo, inicio: comeco, fim: pos, dados };
    } else if (tipo === 1) {
      pos += 8;
      yield { campo, tipo, inicio: comeco, fim: pos };
    } else if (tipo === 5) {
      pos += 4;
      yield { campo, tipo, inicio: comeco, fim: pos };
    } else {
      throw new Error(`tipo protobuf não suportado: ${tipo}`);
    }
  }
  if (pos !== fim) throw new Error("protobuf truncado");
}

function texto(dados) {
  return utf8.decode(dados);
}

function lerParChaveValor(dados) {
  let chave = "";
  let valor = "";
  for (const c of campos(dados)) {
    if (c.campo === 1 && c.tipo === 2) chave = texto(c.dados);
    if (c.campo === 2 && c.tipo === 2) valor = texto(c.dados);
  }
  return [chave, valor];
}

// ValueInfoProto -> {nome, dims}. dims: número, nome simbólico ou null.
function lerValorInfo(dados) {
  let nome = "";
  const dims = [];
  for (const c of campos(dados)) {
    if (c.campo === 1 && c.tipo === 2) nome = texto(c.dados);
    if (c.campo !== 2 || c.tipo !== 2) continue;
    for (const t of campos(c.dados)) {
      if (t.campo !== 1 || t.tipo !== 2) continue; // tensor_type
      for (const s of campos(t.dados)) {
        if (s.campo !== 2 || s.tipo !== 2) continue; // shape
        for (const d of campos(s.dados)) {
          if (d.campo !== 1 || d.tipo !== 2) continue; // dim
          let dim = null;
          for (const v of campos(d.dados)) {
            if (v.campo === 1 && v.tipo === 0) dim = v.valor;
            if (v.campo === 2 && v.tipo === 2) dim = texto(v.dados);
          }
          dims.push(dim);
        }
      }
    }
  }
  return { nome, dims };
}

function lerGrafo(dados) {
  const entradas = [];
  const saidas = [];
  for (const c of campos(dados)) {
    if (c.tipo !== 2) continue;
    if (c.campo === GRAFO_ENTRADA) entradas.push(lerValorInfo(c.dados));
    if (c.campo === GRAFO_SAIDA) saidas.push(lerValorInfo(c.dados));
  }
  return { entradas, saidas };
}

// Lê tudo que o leitor precisa do .onnx: metadados, passaporte, entradas e saídas do grafo.
export function inspecionar(bytes) {
  if (!(bytes instanceof Uint8Array)) bytes = new Uint8Array(bytes);
  const metadados = {};
  let produtor = "";
  let grafo = { entradas: [], saidas: [] };
  for (const c of campos(bytes)) {
    if (c.tipo !== 2) continue;
    if (c.campo === CAMPO_PRODUTOR) produtor = texto(c.dados);
    if (c.campo === CAMPO_GRAFO) grafo = lerGrafo(c.dados);
    if (c.campo === CAMPO_METADADOS) {
      const [chave, valor] = lerParChaveValor(c.dados);
      metadados[chave] = valor; // a última ocorrência vence
    }
  }
  let passaporte = null;
  let erroPassaporte = null;
  if (CHAVE in metadados) {
    try {
      passaporte = JSON.parse(metadados[CHAVE]);
    } catch (e) {
      erroPassaporte = `JSON inválido em ${CHAVE}: ${e.message}`;
    }
  }
  return { produtor, metadados, passaporte, erroPassaporte, ...grafo };
}

function varint(n) {
  const saida = [];
  while (n >= 128) {
    saida.push((n % 128) | 0x80);
    n = Math.floor(n / 128);
  }
  saida.push(n);
  return saida;
}

function campoTexto(numero, str) {
  const b = codificador.encode(str);
  return [...varint(numero * 8 + 2), ...varint(b.length), ...b];
}

// Devolve um novo .onnx com o passaporte gravado. Remove um passaporte anterior para não duplicar
// a chave (o onnx.checker rejeita chaves repetidas). O restante do arquivo é copiado byte a byte.
export function carimbar(bytes, passaporte) {
  if (!(bytes instanceof Uint8Array)) bytes = new Uint8Array(bytes);
  const partes = [];
  let total = 0;
  for (const c of campos(bytes)) {
    if (c.campo === CAMPO_METADADOS && c.tipo === 2 && lerParChaveValor(c.dados)[0] === CHAVE) {
      continue;
    }
    const parte = bytes.subarray(c.inicio, c.fim);
    partes.push(parte);
    total += parte.length;
  }
  const par = [...campoTexto(1, CHAVE), ...campoTexto(2, JSON.stringify(passaporte))];
  const novo = Uint8Array.from([...varint(CAMPO_METADADOS * 8 + 2), ...varint(par.length), ...par]);
  const saida = new Uint8Array(total + novo.length);
  let pos = 0;
  for (const parte of partes) {
    saida.set(parte, pos);
    pos += parte.length;
  }
  saida.set(novo, pos);
  return saida;
}

// Checagens semânticas que o formulário e o leitor precisam antes de rodar. Espelha o essencial
// de passaporte/carimbar.py; devolve a lista de problemas (vazia = ok).
export function problemas(p, grafo = null) {
  const erros = [];
  if (!p || typeof p !== "object") return ["passaporte ausente"];
  if (p.versao !== 1) erros.push(`versao ${p.versao} não suportada (esperado 1)`);
  if (!["deteccao", "classificacao"].includes(p.tarefa)) erros.push(`tarefa inválida: ${p.tarefa}`);
  if (!Array.isArray(p.classes) || p.classes.length === 0) erros.push("classes vazias");
  const e = p.entrada ?? {};
  if (!["NCHW", "NHWC"].includes(e.layout)) erros.push(`entrada.layout inválido: ${e.layout}`);
  if (!Array.isArray(e.tamanho) || e.tamanho.length !== 2) erros.push("entrada.tamanho deve ser [altura, largura]");
  if (!["letterbox", "esticar", "recorte_central"].includes(e.redimensionar)) {
    erros.push(`entrada.redimensionar inválido: ${e.redimensionar}`);
  }
  const formatos = ["caixas_xywh_por_classe", "caixas_xyxy_conf_classe", "probabilidades", "logits"];
  if (!formatos.includes(p.saida?.formato)) erros.push(`saida.formato inválido: ${p.saida?.formato}`);
  if (grafo) {
    if (!grafo.entradas.some((x) => x.nome === e.tensor)) erros.push(`entrada.tensor '${e.tensor}' não existe no grafo`);
    if (!grafo.saidas.some((x) => x.nome === p.saida?.tensor)) erros.push(`saida.tensor '${p.saida?.tensor}' não existe no grafo`);
  }
  if (p.decisao) {
    const fora = p.decisao.alertar_se.filter((c) => !p.classes.includes(c));
    if (fora.length) erros.push(`decisao.alertar_se fora das classes: ${fora.join(", ")}`);
  }
  return erros;
}
