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

Quem lê o modelo não precisa saber de YOLO, Keras ou PyTorch: basta seguir o que o passaporte manda.
Um arquivo, sem txt ao lado, e o mesmo leitor serve para qualquer modelo carimbado.

> **Para usar:** acesse **[yolonnx.vercel.app](https://yolonnx.vercel.app)**, carregue um `.onnx` com
> passaporte e uma imagem. A inferência roda no próprio navegador.

## 🎯 O produto

**Decisão automatizada:** numa foto de obra, existe trabalhador **sem capacete**?

| estado | quando | o que acontece |
| --- | --- | --- |
| **ALERTA** | alguma cabeça sem capacete com confiança ≥ `limiar_alerta` | aviso com a quantidade e a maior confiança |
| **VERIFICAR** | a melhor candidata cai dentro de `zona_incerta` | um humano confere |
| **OK** | nada relevante acima da zona incerta | nenhum aviso |
| **INCERTO** | (classificação) top 1 abaixo da confiança mínima | mostra o top 3, não decide |

A regra fica no passaporte, não no leitor: trocar o limiar é carimbar de novo, sem mexer em código.

**Fallback de contrato:** um `.onnx` sem passaporte não é adivinhado. O leitor propõe um rascunho a
partir dos tensores do grafo, valida o que for completado e devolve o `.onnx` carimbado.

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
| `web/` | o leitor no navegador, publicado em [yolonnx.vercel.app](https://yolonnx.vercel.app) |
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

## ▶️ Como reproduzir

Pré-requisito: [uv](https://docs.astral.sh/uv/). Na raiz do workspace das entregas
(`Entregas - Matheus Marinho/`):

```bash
uv sync --all-packages
cd entrega-03-onnx
uv run pytest -q                 # carimbo e validação do passaporte
node --test tests/*.test.mjs     # leitura do protobuf, decodificadores e decisão (Node.js 20+)

# modelo de demonstração: YOLO11n pré-treinado, classes da Aula 13
uv run python -m passaporte.de_ultralytics --pesos yolo11n.pt \
  --classes treino/classes_coco80_pt.txt --nome "YOLO11n COCO (português)" \
  --destino modelos/yolo11n_pt.onnx \
  --alertar-se pessoa --limiar-alerta 0.5 --zona-incerta 0.3 0.5 --mensagem "Pessoa detectada"

# ver o passaporte de qualquer .onnx
uv run python -m passaporte.carimbar mostrar modelos/yolo11n_pt.onnx
```

Sem `--pesos`, `de_ultralytics.py` usa o `runs/detect/*/weights/best.pt` mais recente. O `.onnx`
gerado pode ser aberto direto em [yolonnx.vercel.app](https://yolonnx.vercel.app).

### ☁️ Google Colab (treino no Roboflow)

1. Abra `treino/treino_hardhat_colab.ipynb` no Colab com **GPU T4**.
2. Em 🔑 *Secrets*, cadastre `ROBOFLOW_API_KEY` e permita o acesso do notebook. A chave não aparece no
   código.
3. **Ambiente de execução → Executar tudo**. No fim, `capacete_best.onnx` é baixado.
4. Copie o arquivo para `modelos/` e abra em [yolonnx.vercel.app](https://yolonnx.vercel.app).

## ✅ Verificação feita

- **Fluxo do best:** treino curto de 3 épocas no `coco8`. O `best.pt` escolhido foi o da época 3
  (mAP50-95 0,4407, maior que 0,4094 das épocas 1 e 2, apesar do mAP50 menor), e o passaporte
  registrou exatamente esses valores do `results.csv`.
- **Paridade com o Ultralytics:** `bus.jpg` com o YOLO11n carimbado. O leitor e o
  `YOLO("yolo11n_pt.onnx").predict()` do Python deram as mesmas 5 detecções (ônibus 94%, pessoas
  90%, 85%, 83%, 40%), com caixas iguais a até 1 px, e a decisão foi **ALERTA** (3 pessoas acima de
  50%).
- **Fallback:** para a CNN sem passaporte, o rascunho saiu do grafo (entrada `imagem [1, 3, 128, 128]`,
  saída `logits [1, 2]`); uma regra com classe inexistente foi recusada; depois de corrigido, o modelo
  foi carimbado. Com pesos aleatórios o top 1 ficou em 55%, e o leitor respondeu
  **INCERTO** em vez de chutar "gato".
- **Compatibilidade:** um `.onnx` carimbado pelo leitor passa no `onnx.checker` e é lido pelo
  `carimbar.py`; carimbar de novo substitui a chave em vez de duplicar.

## ⚠️ Limitações

- A v1 cobre detecção e classificação. Segmentação, pose e OBB ficam como formatos futuros.
- O passaporte é convenção deste projeto. Outras ferramentas ignoram a chave, sem quebrar nada, mas
  também não a aproveitam.
- O leitor redimensiona a imagem de forma diferente do OpenCV usado pelo Ultralytics. A diferença foi
  de até 1 px no teste feito, mas pode crescer em imagens muito pequenas.
- Os limiares do bloco `decisao` no notebook (0,5 e zona 0,3 a 0,5) são um ponto de partida, não
  foram calibrados contra o custo de um falso negativo na obra.

## 📚 Referências

- Material da Aula 13: `materiais/6-onnx-Aula 13/`.
- [Roboflow Universe, Hard Hat Workers](https://universe.roboflow.com/joseph-nelson/hard-hat-workers).
- [ONNX, `onnx.proto`](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto): `ModelProto.metadata_props` é o campo 14.
- [Ultralytics, exportação para ONNX](https://docs.ultralytics.com/integrations/onnx/).
