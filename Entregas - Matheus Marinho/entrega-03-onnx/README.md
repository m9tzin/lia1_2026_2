# 🛂 Passaporte ONNX · [yolonnx.vercel.app](https://yolonnx.vercel.app)

*Um leitor universal de modelos `.onnx`, guiado por contrato.*

**Aluno:** Matheus Sousa Marinho · **Matrícula:** 202206132 · **Disciplina:** LIA 1 (2026/2)

## 🧭 Em uma frase

Um YOLO11n treinado num dataset do **Roboflow** detecta **buracos na via**. Ele é exportado para
`.onnx` com um **passaporte** gravado dentro do arquivo e roda em
**[yolonnx.vercel.app](https://yolonnx.vercel.app)**, um leitor que abre qualquer `.onnx` carimbado.
O site já abre com esse modelo e uma foto de rua carregados, mostrando a decisão.

```
Roboflow: pothole-detection-yolo-v8 (fotos de rua com os buracos anotados)
   ↓  treino no Colab, GPU T4            treino/treino_buracos_colab.ipynb
best.pt: época 36 de 50 (mAP50 0,61)
   ↓  exportação + carimbo               passaporte/de_ultralytics.py
buraco_best.onnx: modelo + passaporte no mesmo arquivo
   ↓  pré-carregado no leitor            web/public/exemplo/ → yolonnx.vercel.app
decisão: ALERTA, VERIFICAR ou OK
```

O leitor é genérico: não sabe nada de buracos. Tudo o que ele usa (classes, pré-processamento, regra
de decisão, métricas) vem do passaporte dentro do `.onnx`.

## 1. 🎯 O problema

Numa foto de rua, existe **buraco** no asfalto que precise de manutenção? O sistema responde com um
estado, não com uma lista de caixas:

| estado | quando | o que acontece |
| --- | --- | --- |
| **ALERTA** | algum buraco com confiança ≥ 0,5 (`limiar_alerta`) | aviso com a quantidade e a maior confiança |
| **VERIFICAR** | a melhor candidata fica entre 0,3 e 0,5 (`zona_incerta`) | um humano confere a foto |
| **OK** | nada acima de 0,3 | nenhum aviso |

Os limiares ficam no passaporte do modelo, não no código do leitor. Mudar a política é carimbar o
modelo de novo.

## 2. 🗂️ Dataset (desafio Roboflow)

[pothole-detection-yolo-v8](https://universe.roboflow.com/kartik-zvust/pothole-detection-yolo-v8),
versão 1, no Roboflow Universe: fotos de rua com os buracos anotados numa única classe, `potholes`,
traduzida por nome para `buraco`.

O dataset não está fixo no código. O primeiro bloco do notebook reúne workspace, projeto, versão,
tradução das classes, regra de decisão e parâmetros de treino; o resto do notebook não cita nome de
classe. Antes de treinar, o notebook confere se a regra de alerta cita uma classe que existe no
dataset, para não descobrir o erro só depois de 50 épocas.

## 3. 🏋️ Treino e o best.pt

O notebook `treino/treino_buracos_colab.ipynb` baixa o dataset pela API do Roboflow e treina um
YOLO11n no Colab com GPU T4 (`epochs=50`, `imgsz=640`, `patience=15`, `batch=32`).

Durante o treino, o Ultralytics valida o modelo ao fim de cada época e calcula um *fitness*. Em
`runs/detect/<nome>/weights/` ficam:

- `last.pt`: pesos da última época, útil para retomar o treino;
- `best.pt`: pesos da época de **maior fitness**, que é o que vai para produção.

Na versão 8.4 usada aqui, o fitness de detecção é o **mAP50-95** (pesos `[0, 0, 0, 1]` sobre
P, R, mAP50, mAP50-95 em `ultralytics/utils/metrics.py`). Versões antigas usavam
`0.1·mAP50 + 0.9·mAP50-95`. Por isso o projeto não recalcula nada: lê `train_metrics`, que o próprio
`best.pt` guarda, e encontra a época correspondente no `results.csv`.

Resultado do treino publicado (validação):

| época do best | épocas treinadas | mAP50 | mAP50-95 | precisão | recall |
| --- | --- | --- | --- | --- | --- |
| 36 | 50 | 0,608 | 0,238 | 0,650 | 0,584 |

O `patience=15` não chegou a interromper o treino: a melhor época foi a 36 e as 14 seguintes não a
superaram.

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

Passaporte do `buraco_best.onnx` publicado, copiado de `metadata_props` do próprio arquivo:

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
    "framework": "ultralytics 8.4.159", "pesos": "best.pt", "exportado_em": "2026-09-22",
    "dataset": "roboflow kartik-zvust/pothole-detection-yolo-v8 v1 (pothole-detection-yolo-v8)",
    "best": { "epoca": 36, "epocas_treinadas": 50, "mAP50": 0.6076, "mAP50_95": 0.2379,
              "precisao": 0.6498, "recall": 0.5839, "fitness": 0.2379 }
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

## 5. 🌐 O leitor

Acesse **[yolonnx.vercel.app](https://yolonnx.vercel.app)**. A página abre com o `buraco_best.onnx`
e uma foto de rua já carregados e mostra **ALERTA: Buraco na via (3, maior 82%)**. O modelo roda no
próprio navegador, com ONNX Runtime Web; nenhuma imagem sai do computador de quem usa.

- **Outro modelo ou outra foto:** solte o arquivo na área correspondente. O que você solta tem
  prioridade, mesmo que o exemplo ainda esteja baixando.
- **Confiança e IoU:** os controles laterais começam nos valores do passaporte e só re-decodificam a
  saída, sem rodar o modelo de novo.
- **Modelo sem passaporte:** o leitor não adivinha. Ele propõe um rascunho a partir do grafo, valida
  com as mesmas regras do `carimbar.py` e devolve o arquivo carimbado para download.

### O exemplo pré-carregado

Fica em `web/public/exemplo/`, e `exemplo.json` diz o que abrir:

```json
{
  "modelo": "buraco_best.onnx",
  "imagem": "buracos_newport.jpg",
  "credito": {
    "texto": "Foto: Editor5807, CC BY 3.0, via Wikimedia Commons",
    "url": "https://commons.wikimedia.org/wiki/File:Newport_Whitepit_Lane_pot_holes_2.JPG"
  }
}
```

Trocar o exemplo é trocar arquivos, sem mexer no código. A foto não faz parte do dataset: é
[Newport Whitepit Lane pot holes 2](https://commons.wikimedia.org/wiki/File:Newport_Whitepit_Lane_pot_holes_2.JPG),
de Editor5807, sob [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). O crédito aparece
abaixo do resultado enquanto a foto do exemplo está na tela.

## ▶️ Como reproduzir

### ☁️ Treino no Colab (Roboflow → `buraco_best.onnx`)

1. Crie uma conta em [app.roboflow.com](https://app.roboflow.com) e copie a **Private API Key**
   (*Settings → API Keys*).
2. Abra `treino/treino_buracos_colab.ipynb` no Colab e escolha **GPU T4** em *Ambiente de execução →
   Alterar o tipo de ambiente de execução*.
3. Em 🔑 *Secrets*, cadastre `ROBOFLOW_API_KEY` e ative o acesso do notebook. A chave não aparece no
   código.
4. **Ambiente de execução → Executar tudo**. No fim, o notebook baixa `buraco_best.onnx` e
   `exemplo.jpg`, a foto de teste com o buraco detectado com mais confiança.
5. Guarde o `.onnx` em `modelos/`.

### 🌐 Publicar como exemplo do site

1. Copie o modelo e uma foto para `web/public/exemplo/` e aponte `exemplo.json` para eles. Foto de
   fora do dataset precisa de licença que permita publicar; se a licença pedir crédito, preencha
   `credito`.
2. Na pasta `web/`: `npm install`, `npm run build` e `vercel deploy --prod`.

### 💻 Localmente (testes e modelo de demonstração)

Pré-requisito: [uv](https://docs.astral.sh/uv/). Na raiz do workspace das entregas
(`Entregas - Matheus Marinho/`):

```bash
uv sync --all-packages
cd entrega-03-onnx
uv run pytest -q                 # carimbo e validação do passaporte
node --test tests/*.test.mjs     # leitura do protobuf, decodificadores e decisão (Node.js 20+)

# ver o passaporte de qualquer .onnx
uv run python -m passaporte.carimbar mostrar modelos/buraco_best.onnx

# demonstração sem treinar: YOLO11n pré-treinado no COCO, com as classes em português da Aula 13
uv run python -m passaporte.de_ultralytics --pesos yolo11n.pt \
  --classes treino/classes_coco80_pt.txt --nome "YOLO11n COCO (português)" \
  --destino modelos/yolo11n_pt.onnx \
  --alertar-se pessoa --limiar-alerta 0.5 --zona-incerta 0.3 0.5 --mensagem "Pessoa detectada"

# o leitor em modo de desenvolvimento
cd web && npm install && npm run dev
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
| `modelos/buraco_best.onnx` | o modelo entregue, com passaporte |
| `modelos/yolo11n_pt.onnx` | YOLO11n COCO carimbado, usado nos testes de paridade |
| `web/` | o leitor publicado em [yolonnx.vercel.app](https://yolonnx.vercel.app) (React, Vite, ONNX Runtime Web) |
| `web/public/exemplo/` | modelo, foto e `exemplo.json` que o site abre por padrão |
| `tests/` | 13 testes em Python e 14 em Node |

## ✅ Verificação feita

- **Modelo de buracos em foto nova:** na foto de Newport, que não faz parte do dataset, o
  `YOLO("buraco_best.onnx").predict()` do Python e o leitor encontraram os mesmos 3 buracos (82%,
  77% e 55%, caixas iguais a até 1 px). Os três passam de 0,5, e a decisão foi **ALERTA**.
- **Em produção:** depois do deploy, `yolonnx.vercel.app` serve o modelo (10,6 MB), a foto e o
  `exemplo.json`, e a página abre direto no ALERTA com o crédito da foto.
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
- Os limiares do bloco `decisao` (0,5 e zona 0,3 a 0,5) são um ponto de partida. Não foram
  calibrados contra o custo de um buraco não reportado.
- O exemplo pré-carregado baixa cerca de 11 MB ao abrir a página, mesmo para quem só quer usar o
  próprio `.onnx`.
- A foto do exemplo só é decodificada quando a aba está visível. Aberto numa aba em segundo plano, o
  resultado aparece quando a aba ganha foco.
- A v1 cobre detecção e classificação. Segmentação, pose e OBB ficam como formatos futuros.
- O passaporte é convenção deste projeto. Outras ferramentas ignoram a chave, sem quebrar nada, mas
  também não a aproveitam.
- O leitor redimensiona a imagem de forma diferente do OpenCV usado pelo Ultralytics. A diferença foi
  de até 1 px nos testes feitos, mas pode crescer em imagens muito pequenas.

## 📚 Referências

- Material da Aula 13: `materiais/6-onnx-Aula 13/`.
- [Roboflow Universe, pothole-detection-yolo-v8](https://universe.roboflow.com/kartik-zvust/pothole-detection-yolo-v8).
- [Wikimedia Commons, Newport Whitepit Lane pot holes 2](https://commons.wikimedia.org/wiki/File:Newport_Whitepit_Lane_pot_holes_2.JPG) (CC BY 3.0, Editor5807).
- [ONNX, `onnx.proto`](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto): `ModelProto.metadata_props` é o campo 14.
- [Ultralytics, exportação para ONNX](https://docs.ultralytics.com/integrations/onnx/).
