# 🛂 Passaporte ONNX: um leitor universal guiado por contrato

**Aluno:** Matheus Sousa Marinho · **Matrícula:** 202206132 · **Disciplina:** LIA 1 (2026/2)

Na Aula 13, o YOLO11n é exportado para `.onnx` e as classes viajam num `classes_coco80_pt.txt` à
parte, porque "o ONNX não garante o transporte dos nomes das classes". Este projeto resolve isso de
outro jeito: cada modelo recebe um **passaporte**, um JSON gravado dentro do próprio `.onnx` (em
`metadata_props`, chave `lia.passaporte`), que diz a qualquer leitor:

- quais são as classes e em que ordem;
- como preparar a imagem (tamanho, letterbox ou recorte, escala, média, desvio, RGB ou BGR);
- como interpretar a saída (um vocabulário fechado de formatos);
- qual decisão tomar (limiar de alerta, zona incerta, mensagem);
- de onde o modelo veio, incluindo a época e as métricas do `best.pt`.

O leitor web não sabe nada de YOLO, Keras ou PyTorch. Ele só executa o que o passaporte manda. Um
arquivo, sem txt ao lado, e o mesmo leitor serve para qualquer modelo carimbado.

## 🎯 O produto

**Decisão automatizada:** numa foto de obra, existe trabalhador **sem capacete**?

| estado | quando | o que acontece |
| --- | --- | --- |
| **ALERTA** | alguma cabeça sem capacete com confiança ≥ `limiar_alerta` | aviso com a quantidade e a maior confiança |
| **VERIFICAR** | a melhor candidata cai dentro de `zona_incerta` | caixa tracejada, um humano confere |
| **OK** | nada relevante acima da zona incerta | nenhum aviso |
| **INCERTO** | (classificação) top 1 abaixo da confiança mínima | mostra o top 3, não decide |

A regra fica no passaporte, não no leitor: trocar o limiar é carimbar de novo, sem mexer em código.

**Fallback de contrato:** um `.onnx` sem passaporte não é adivinhado. O leitor mostra os tensores do
grafo, abre um rascunho do passaporte já preenchido com o que dá para ler do arquivo, valida o que
você completar e devolve o `.onnx` carimbado para download.

## 🗂️ Dataset (desafio Roboflow)

[Hard Hat Workers](https://universe.roboflow.com/joseph-nelson/hard-hat-workers), no Roboflow
Universe: cerca de 7 mil imagens de canteiros de obra anotadas com `head`, `helmet` e `person`. O
treino usa a versão `raw_HeadHelmetClasses` (só cabeça e capacete), porque `person` não entra na
decisão. As classes são traduzidas por nome: `head → sem capacete`, `helmet → capacete`.

## 🏅 O que é o best.pt

Durante `model.train(...)` o Ultralytics valida o modelo ao fim de cada época e calcula um
*fitness*. Em `runs/detect/<nome>/weights/` ficam:

- `last.pt`: pesos da última época, útil para retomar o treino;
- `best.pt`: pesos da época de **maior fitness**, que é o que vai para produção.

Na versão 8.4 instalada aqui, o fitness de detecção é o **mAP50-95** (pesos `[0, 0, 0, 1]` sobre
P, R, mAP50, mAP50-95 em `ultralytics/utils/metrics.py`). Versões antigas usavam
`0.1·mAP50 + 0.9·mAP50-95`. Por isso o script não recalcula nada: lê `train_metrics`, que o próprio
`best.pt` guarda, e encontra a época correspondente no `results.csv`. O caminho do arquivo também
fica em `model.trainer.best` logo após o treino.

## 📁 Estrutura

| arquivo | finalidade |
| --- | --- |
| `passaporte/esquema_v1.json` | o contrato (JSON Schema) |
| `passaporte/carimbar.py` | grava, lê e valida o passaporte contra o esquema e contra o grafo |
| `passaporte/de_ultralytics.py` | `best.pt` → ONNX com passaporte preenchido automaticamente |
| `passaporte/de_pytorch.py` | CNN PyTorch comum carimbada à mão, prova de que o leitor não depende do Ultralytics |
| `treino/treino_hardhat_colab.ipynb` | Roboflow → YOLO11n → `best.pt` → `capacete_best.onnx` (Colab, GPU) |
| `treino/classes_coco80_pt.txt` | as 80 classes COCO em português da Aula 13, para carimbar o `yolo11n.pt` |
| `web/` | o leitor universal: Vite + React + TypeScript + Tailwind v4, com componentes do Magic UI |
| `web/src/lib/passaporte.js` | leitura e gravação do passaporte direto nos bytes do protobuf |
| `web/src/lib/decodificadores.js` | pré-processamento, NMS e um decodificador por formato de saída |
| `web/src/lib/motor.ts` | sessão do ONNX Runtime Web (wasm empacotado, funciona offline) e rascunho de passaporte |
| `tests/` | 13 testes em Python e 14 em Node |
| `modelos/` | destino do `capacete_best.onnx` baixado do Colab |

## 📜 O contrato

Exemplo do passaporte do modelo de capacete (os números de `best` saem do treino real):

```json
{
  "versao": 1,
  "nome": "Capacete em obra (YOLO11n)",
  "tarefa": "deteccao",
  "classes": ["sem capacete", "capacete"],
  "entrada": {
    "tensor": "images", "layout": "NCHW", "tamanho": [640, 640], "cores": "RGB",
    "redimensionar": "letterbox", "preenchimento": 114,
    "escala": 0.00392156862745098, "media": [0, 0, 0], "desvio": [1, 1, 1]
  },
  "saida": { "tensor": "output0", "formato": "caixas_xywh_por_classe" },
  "operacao": { "confianca_minima": 0.35, "iou_nms": 0.45 },
  "decisao": {
    "alertar_se": ["sem capacete"], "limiar_alerta": 0.5,
    "zona_incerta": [0.3, 0.5], "mensagem": "Trabalhador sem capacete"
  },
  "origem": {
    "framework": "ultralytics 8.4.x", "pesos": "best.pt",
    "dataset": "roboflow joseph-nelson/hard-hat-workers",
    "best": { "epoca": 0, "mAP50": 0.0, "mAP50_95": 0.0, "precisao": 0.0, "recall": 0.0 }
  }
}
```

Formatos de saída suportados:

| formato | shape | de onde vem |
| --- | --- | --- |
| `caixas_xywh_por_classe` | `[1, 4+nc, N]` | YOLOv8 / YOLO11 (âncoras cruas, o leitor faz o NMS) |
| `caixas_xyxy_conf_classe` | `[1, K, 6]` | YOLO26 / YOLOv10 (já filtrado, sem NMS) |
| `probabilidades` | `[1, nc]` | classificadores com softmax no grafo |
| `logits` | `[1, nc]` | classificadores sem softmax (o leitor aplica) |

Novo formato é um novo decodificador em `web/src/lib/decodificadores.js`; o resto não muda.

Ao carimbar, `carimbar.py` recusa o passaporte se: o JSON não segue o esquema, o tensor declarado não
existe no grafo, o tamanho diverge do input estático, o nº de classes não bate com a saída (por
exemplo 3 classes numa saída com 4 + 2 canais), há classes repetidas ou a regra de alerta cita uma
classe que não existe.

## ▶️ Como executar

### 💻 Localmente

Na raiz do workspace das entregas:

```bash
uv sync --all-packages
cd entrega-03-onnx
uv run pytest -q                 # Python: carimbo e validação
node --test tests/*.test.mjs     # JavaScript: protobuf, decodificadores e decisão

# modelo de demonstração: YOLO11n pré-treinado, classes da Aula 13
uv run python -m passaporte.de_ultralytics --pesos yolo11n.pt \
  --classes treino/classes_coco80_pt.txt --nome "YOLO11n COCO (português)" \
  --destino modelos/yolo11n_pt.onnx \
  --alertar-se pessoa --limiar-alerta 0.5 --zona-incerta 0.3 0.5 --mensagem "Pessoa detectada"

# ver o passaporte de qualquer .onnx
uv run python -m passaporte.carimbar mostrar modelos/yolo11n_pt.onnx

# o leitor
cd web
npm install
npm run dev        # abrir o endereço que o Vite mostrar
npm run build      # versão estática em web/dist
```

Sem `--pesos`, `de_ultralytics.py` usa o `runs/detect/*/weights/best.pt` mais recente.

### ☁️ Google Colab (treino no Roboflow)

1. Abra `treino/treino_hardhat_colab.ipynb` no Colab com **GPU T4**.
2. Em 🔑 *Secrets*, cadastre `ROBOFLOW_API_KEY` e permita o acesso do notebook. A chave não aparece no
   código.
3. **Ambiente de execução → Executar tudo**. No fim, `capacete_best.onnx` é baixado.
4. Copie o arquivo para `modelos/` e abra no leitor.

## 🎨 Interface

Visual inspirado no [Infisical](https://infisical.com/): fundo branco, texto quase preto, bordas de 1 px,
cantos retos, botões em pílula, amarelo neon (`#f7fe62`) como único destaque e rótulos em mono
maiúsculo (JetBrains Mono) com tags entre colchetes, como `[carregado]`. Tema escuro automático pelo
sistema, com botão para alternar.

Componentes do [Magic UI](https://magicui.design/), instalados pelo CLI do shadcn:

| componente | onde |
| --- | --- |
| Grid Pattern | grade do topo, com quadrados neon acesos |
| Dot Pattern | aparece nas áreas de soltar arquivo ao passar o mouse ou arrastar |
| Animated Shiny Text | selo `lia.passaporte v1` |
| Typing Animation | o que o arquivo informa: classes, pré-processamento, regra, métricas |
| Number Ticker | mAP50, mAP50-95 e época do best; tempo de inferência |
| Border Beam | moldura da imagem enquanto a inferência roda |
| Animated List | detecções entrando uma a uma, da maior para a menor confiança |
| Shimmer Button | "Carimbar e baixar" no fluxo sem passaporte |

Nas faixas laterais, fora da coluna central, roda o fundo ASCII
[Fluid](https://asciify.org/docs/backgrounds/fluid) do asciify-engine (MIT), com um rastro que segue
o cursor. A coluna central é sólida e cobre a animação, então nada do conteúdo fica por cima do
ASCII. Os caracteres usam uma cor só, bem perto do fundo: `#1e1e1e` no tema escuro e `#e9e9e7` no
claro. O template copiado fica em `web/src/components/ascii/`, sem alterações.

Caixas desenhadas em amarelo neon; vermelho fica reservado para o que de fato dispara o alerta
(classe de alerta acima de `limiar_alerta`), para que o destaque signifique decisão e não só classe.

## ✅ Verificação feita

- **Fluxo do best:** treino curto de 3 épocas no `coco8`. O `best.pt` escolhido foi o da época 3
  (mAP50-95 0,4407, maior que 0,4094 das épocas 1 e 2, apesar do mAP50 menor), e o passaporte
  registrou exatamente esses valores do `results.csv`.
- **Paridade no navegador:** `bus.jpg` com o YOLO11n carimbado. O leitor web e o
  `YOLO("yolo11n_pt.onnx").predict()` do Python deram as mesmas 5 detecções (ônibus 94%, pessoas
  90%, 85%, 83%, 40%), com caixas iguais a até 1 px. Selo: **ALERTA**, "Pessoa detectada (3, maior
  90%)". Inferência de 152 ms no wasm. Repetido na versão React: mesmas detecções, 135 ms.
- **Fallback:** a CNN sem passaporte abriu o rascunho (entrada `imagem [1, 3, 128, 128]`, saída
  `logits [1, 2]`); uma regra com classe inexistente foi recusada; depois de corrigido, o modelo foi
  carimbado e carregado. Com pesos aleatórios o top 1 ficou em 55%, e o leitor respondeu
  **INCERTO** em vez de chutar "gato".
- **Compatibilidade:** um `.onnx` carimbado pelo JavaScript passa no `onnx.checker` e é lido pelo
  `carimbar.py`; carimbar de novo substitui a chave em vez de duplicar.

## ⚠️ Limitações

- A v1 cobre detecção e classificação. Segmentação, pose e OBB ficam como formatos futuros.
- O passaporte é convenção deste projeto. Outras ferramentas ignoram a chave, sem quebrar nada, mas
  também não a aproveitam.
- O leitor redimensiona com o `canvas` do navegador, e o Ultralytics com o OpenCV. A diferença foi de
  até 1 px e décimos de ponto percentual na confiança no teste feito, mas pode crescer em imagens
  muito pequenas.
- O backend wasm roda em CPU. Para vídeo em tempo real seria preciso WebGPU.
- Os limiares do bloco `decisao` no notebook (0,5 e zona 0,3 a 0,5) são um ponto de partida, não
  foram calibrados contra o custo de um falso negativo na obra.

## 📚 Referências

- Material da Aula 13: `materiais/6-onnx-Aula 13/`.
- [Roboflow Universe, Hard Hat Workers](https://universe.roboflow.com/joseph-nelson/hard-hat-workers).
- [ONNX, `onnx.proto`](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto): `ModelProto.metadata_props` é o campo 14.
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).
- [Ultralytics, exportação para ONNX](https://docs.ultralytics.com/integrations/onnx/).
