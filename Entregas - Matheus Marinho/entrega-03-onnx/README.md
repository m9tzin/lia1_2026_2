# 🛂 Passaporte ONNX · [yolonnx.vercel.app](https://yolonnx.vercel.app)

*Um leitor universal de modelos `.onnx`, guiado por contrato.*

**Aluno:** Matheus Sousa Marinho · **Matrícula:** 202206132 · **Disciplina:** LIA 1 (2026/2)

## 🧭 Em uma frase

Um modelo YOLO11n é treinado num dataset do **Roboflow** para detectar **buracos na via**,
exportado para `.onnx` com um **passaporte** gravado dentro do arquivo, e aberto em
**[yolonnx.vercel.app](https://yolonnx.vercel.app)**, um leitor que roda qualquer `.onnx` carimbado.

```
Roboflow: dataset de buracos no asfalto (fotos de rua com os buracos anotados)
   ↓  treino no Colab                    treino/treino_buracos_colab.ipynb
best.pt: pesos da melhor época do YOLO11n
   ↓  exportação + carimbo               passaporte/de_ultralytics.py
buraco_best.onnx: modelo + passaporte no mesmo arquivo
   ↓  abrir no leitor                    yolonnx.vercel.app (já vem pré-carregado)
decisão: ALERTA, VERIFICAR ou OK
```

O site é genérico: lê qualquer `.onnx` que tenha passaporte. O modelo de buracos é a aplicação que
este projeto entrega e o exemplo que o site abre por padrão.

## 1. 🎯 O problema

Numa foto de rua, existe **buraco** no asfalto que precise de manutenção? O sistema responde com um
estado, não com uma lista de caixas:

| estado | quando | o que acontece |
| --- | --- | --- |
| **ALERTA** | algum buraco com confiança ≥ `limiar_alerta` | aviso com a quantidade e a maior confiança |
| **VERIFICAR** | a melhor candidata cai dentro de `zona_incerta` | um humano confere |
| **OK** | nada relevante acima da zona incerta | nenhum aviso |

Os limiares ficam no passaporte do modelo, não no código do leitor. Mudar a política é carimbar o
modelo de novo.

## 2. 🗂️ Dataset (desafio Roboflow)

[pothole-detection-yolo-v8](https://universe.roboflow.com/kartik-zvust/pothole-detection-yolo-v8),
versão 1, no Roboflow Universe: fotos de rua com os buracos anotados. A classe é traduzida por nome,
`pothole → buraco`.

O dataset não está fixo no código. O primeiro bloco do notebook reúne workspace, projeto, versão,
tradução das classes e regra de decisão; o resto do notebook não cita nome de classe. Antes de treinar,
o notebook confere se a regra de alerta cita uma classe que existe no dataset.

## 3. 🏋️ Treino e o best.pt

O notebook `treino/treino_buracos_colab.ipynb` baixa o dataset pela API do Roboflow e treina um
YOLO11n (`epochs=50`, `imgsz=640`, `patience=15`).

Durante o treino, o Ultralytics valida o modelo ao fim de cada época e calcula um *fitness*. Em
`runs/detect/<nome>/weights/` ficam:

- `last.pt`: pesos da última época, útil para retomar o treino;
- `best.pt`: pesos da época de **maior fitness**, que é o que vai para produção.

Na versão 8.4 usada aqui, o fitness de detecção é o **mAP50-95** (pesos `[0, 0, 0, 1]` sobre
P, R, mAP50, mAP50-95 em `ultralytics/utils/metrics.py`). Versões antigas usavam
`0.1·mAP50 + 0.9·mAP50-95`. Por isso o projeto não recalcula nada: lê `train_metrics`, que o próprio
`best.pt` guarda, e encontra a época correspondente no `results.csv`.

## 4. 🛂 O passaporte

Na Aula 13, as classes viajam num `classes_coco80_pt.txt` à parte, porque "o ONNX não garante o
transporte dos nomes das classes". Aqui, `passaporte/de_ultralytics.py` exporta o `best.pt` para
`.onnx` (mesmos parâmetros da aula) e grava dentro do arquivo, em `metadata_props` na chave
`lia.passaporte`, um JSON que diz a qualquer leitor:

- quais são as classes e em que ordem;
- como preparar a imagem (tamanho, letterbox ou recorte, escala, média, desvio, RGB ou BGR);
- como interpretar a saída;
- qual decisão tomar (limiar de alerta, zona incerta, mensagem);
- de onde o modelo veio, incluindo a época e as métricas do `best.pt`.

Passaporte do `buraco_best.onnx` publicado (os números de `best` saem do treino real):

```json
{
  "versao": 1,
  "nome": "Buraco na via (YOLO11n)",
  "tarefa": "deteccao",
  "classes": ["buraco"],
  "entrada": {
    "tensor": "images", "layout": "NCHW", "tamanho": [640, 640], "cores": "RGB",
    "redimensionar": "letterbox", "preenchimento": 114,
    "escala": 0.00392156862745098, "media": [0, 0, 0], "desvio": [1, 1, 1]
  },
  "saida": { "tensor": "output0", "formato": "caixas_xywh_por_classe" },
  "operacao": { "confianca_minima": 0.35, "iou_nms": 0.45 },
  "decisao": {
    "alertar_se": ["buraco"], "limiar_alerta": 0.5,
    "zona_incerta": [0.3, 0.5], "mensagem": "Buraco na via"
  },
  "origem": {
    "framework": "ultralytics 8.4.159", "pesos": "best.pt",
    "dataset": "roboflow kartik-zvust/pothole-detection-yolo-v8 v1",
    "best": { "epoca": 36, "epocas_treinadas": 50, "mAP50": 0.6076, "mAP50_95": 0.2379,
              "precisao": 0.6498, "recall": 0.5839 }
  }
}
```

O contrato está em `passaporte/esquema_v1.json`. Ao carimbar, `passaporte/carimbar.py` recusa o
passaporte se o JSON não segue o esquema, o tensor declarado não existe no grafo, o tamanho diverge
do input, o nº de classes não bate com a saída, há classes repetidas ou a regra de alerta cita uma
classe que não existe.

Formatos de saída suportados, o que permite carimbar modelos que não vêm do Ultralytics:

| formato | shape | de onde vem |
| --- | --- | --- |
| `caixas_xywh_por_classe` | `[1, 4+nc, N]` | YOLOv8 / YOLO11 |
| `caixas_xyxy_conf_classe` | `[1, K, 6]` | YOLO26 / YOLOv10 |
| `probabilidades` | `[1, nc]` | classificadores com softmax no grafo |
| `logits` | `[1, nc]` | classificadores sem softmax |

## 5. 🌐 Uso

Acesse **[yolonnx.vercel.app](https://yolonnx.vercel.app)**. A página já abre com um modelo e uma foto
de exemplo carregados e mostra o estado. Solte outro `.onnx` ou outra foto para substituir; o que você
solta tem prioridade, mesmo que o exemplo ainda esteja baixando. O leitor lê o passaporte e roda o
modelo no próprio navegador.

O exemplo fica em `web/public/exemplo/`: `exemplo.json` aponta o modelo e a foto, então trocar o
exemplo é trocar arquivos, sem mexer no código. O exemplo publicado é o `buraco_best.onnx` com uma
foto de rua que o modelo nunca viu no treino:
[Newport Whitepit Lane pot holes 2](https://commons.wikimedia.org/wiki/File:Newport_Whitepit_Lane_pot_holes_2.JPG),
de Editor5807, licença [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), via Wikimedia
Commons. O crédito vem do campo `credito` do `exemplo.json` e aparece abaixo do resultado. Um `.onnx` sem
passaporte não é adivinhado: o leitor propõe um rascunho a partir do grafo e devolve o arquivo
carimbado depois de validado.

## ▶️ Como reproduzir

### ☁️ Treino no Colab (Roboflow → `buraco_best.onnx`)

1. Abra `treino/treino_buracos_colab.ipynb` no Colab com **GPU T4**.
2. Em 🔑 *Secrets*, cadastre `ROBOFLOW_API_KEY` e permita o acesso do notebook. A chave não aparece no
   código.
3. **Ambiente de execução → Executar tudo**. No fim, o notebook baixa `buraco_best.onnx` e
   `exemplo.jpg`, a foto de teste com o buraco detectado com mais confiança.
4. Guarde o `.onnx` em `modelos/`. Para ele virar o exemplo do site, copie o modelo e uma foto para
   `web/public/exemplo/`, aponte `exemplo.json` para eles e publique o site de novo. Foto de fora do
   dataset precisa de licença que permita publicar; se a licença pedir crédito, preencha `credito`.

### 💻 Localmente (testes e modelo de demonstração)

Pré-requisito: [uv](https://docs.astral.sh/uv/). Na raiz do workspace das entregas
(`Entregas - Matheus Marinho/`):

```bash
uv sync --all-packages
cd entrega-03-onnx
uv run pytest -q                 # carimbo e validação do passaporte
node --test tests/*.test.mjs     # leitura do protobuf, decodificadores e decisão (Node.js 20+)

# demonstração sem treinar: YOLO11n pré-treinado no COCO, com as classes em português da Aula 13
uv run python -m passaporte.de_ultralytics --pesos yolo11n.pt \
  --classes treino/classes_coco80_pt.txt --nome "YOLO11n COCO (português)" \
  --destino modelos/yolo11n_pt.onnx \
  --alertar-se pessoa --limiar-alerta 0.5 --zona-incerta 0.3 0.5 --mensagem "Pessoa detectada"

# ver o passaporte de qualquer .onnx
uv run python -m passaporte.carimbar mostrar modelos/yolo11n_pt.onnx
```

## 📁 Estrutura

| arquivo | finalidade |
| --- | --- |
| `treino/treino_buracos_colab.ipynb` | Roboflow → YOLO11n → `best.pt` → `buraco_best.onnx` + foto de exemplo (Colab, GPU) |
| `passaporte/de_ultralytics.py` | `best.pt` → `.onnx` com passaporte preenchido automaticamente |
| `passaporte/carimbar.py` | grava, lê e valida o passaporte contra o esquema e contra o grafo |
| `passaporte/esquema_v1.json` | o contrato (JSON Schema) |
| `passaporte/de_pytorch.py` | CNN PyTorch comum carimbada à mão, prova de que o leitor não depende do Ultralytics |
| `treino/classes_coco80_pt.txt` | as 80 classes COCO em português da Aula 13, para o modelo de demonstração |
| `modelos/` | destino do `buraco_best.onnx` |
| `web/` | o leitor publicado em [yolonnx.vercel.app](https://yolonnx.vercel.app) |
| `web/public/exemplo/` | modelo e foto que o site abre por padrão, apontados por `exemplo.json` |
| `tests/` | 13 testes em Python e 14 em Node |

## ✅ Verificação feita

- **Modelo de buracos:** 50 épocas no Colab (T4); o `best.pt` é o da época 36, com mAP50 0,61,
  mAP50-95 0,24, precisão 0,65 e recall 0,58 na validação. Na foto de Newport, que não faz parte do
  dataset, o `YOLO(...).predict()` do Python e o leitor encontraram os mesmos 3 buracos (82%, 77% e
  55%, caixas iguais a até 1 px), e a decisão foi **ALERTA**.
- **Fluxo do best:** treino curto de 3 épocas no `coco8`. O `best.pt` escolhido foi o da época 3
  (mAP50-95 0,4407, maior que 0,4094 das épocas 1 e 2, apesar do mAP50 menor), e o passaporte
  registrou exatamente esses valores do `results.csv`.
- **Paridade com o Ultralytics:** `bus.jpg` com o YOLO11n carimbado. O leitor e o
  `YOLO("yolo11n_pt.onnx").predict()` do Python deram as mesmas 5 detecções (ônibus 94%, pessoas
  90%, 85%, 83%, 40%), com caixas iguais a até 1 px, e a decisão foi **ALERTA** (3 pessoas acima de
  50%).
- **Modelo sem passaporte:** para a CNN sem passaporte, o rascunho saiu do grafo (entrada
  `imagem [1, 3, 128, 128]`, saída `logits [1, 2]`); uma regra com classe inexistente foi recusada;
  depois de corrigido, o modelo foi carimbado. Com pesos aleatórios o top 1 ficou em 55%, e o leitor
  respondeu **INCERTO** em vez de chutar "gato".
- **Compatibilidade:** um `.onnx` carimbado pelo leitor passa no `onnx.checker` e é lido pelo
  `carimbar.py`; carimbar de novo substitui a chave em vez de duplicar.

## ⚠️ Limitações

- O recall de 0,58 na validação quer dizer que cerca de 4 em cada 10 buracos anotados passam
  despercebidos. Para triagem de manutenção serve como apoio; como fonte única, não.
- O exemplo pré-carregado baixa cerca de 10 MB ao abrir a página, mesmo para quem só quer usar o
  próprio `.onnx`.
- A v1 cobre detecção e classificação. Segmentação, pose e OBB ficam como formatos futuros.
- O passaporte é convenção deste projeto. Outras ferramentas ignoram a chave, sem quebrar nada, mas
  também não a aproveitam.
- O leitor redimensiona a imagem de forma diferente do OpenCV usado pelo Ultralytics. A diferença foi
  de até 1 px no teste feito, mas pode crescer em imagens muito pequenas.
- Os limiares do bloco `decisao` (0,5 e zona 0,3 a 0,5) são um ponto de partida, não foram
  calibrados contra o custo de um falso negativo na obra.

## 📚 Referências

- Material da Aula 13: `materiais/6-onnx-Aula 13/`.
- [Roboflow Universe, pothole-detection-yolo-v8](https://universe.roboflow.com/kartik-zvust/pothole-detection-yolo-v8).
- [ONNX, `onnx.proto`](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto): `ModelProto.metadata_props` é o campo 14.
- [Ultralytics, exportação para ONNX](https://docs.ultralytics.com/integrations/onnx/).
