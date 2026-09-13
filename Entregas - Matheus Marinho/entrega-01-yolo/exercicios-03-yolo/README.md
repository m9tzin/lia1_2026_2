# MVP de Monitoramento Visual de Foco

**Aluno:** Matheus Sousa Marinho · **Matrícula:** 202206132 · **Disciplina:** LIA 1 (2026/2)

Esta entrega evolui o exercício de YOLO26 para um MVP executado inteiramente em um
notebook Google Colab. Ele recebe um vídeo gravado de uma sessão de estudo e usa
heurísticas visuais para estimar períodos focados, distraídos, ausentes e sem evidência
suficiente (`UNKNOWN`). O notebook:

- reaproveita o modelo pré-treinado `yolo26n.pt` para `person` e `cell phone`;
- usa MediaPipe Face Landmarker para orientação aproximada da cabeça;
- mede o detector no dataset COCO128 da Ultralytics antes de processar o vídeo;
- valida YOLO, rosto, regras e debounce temporal antes da integração;
- processa um arquivo `.mp4` sem carregar todos os frames na memória;
- produz vídeo anotado, episódios, métricas finais e gráficos simples;
- concentra todos os parâmetros ajustáveis em uma única célula.

O resultado é uma estimativa experimental, não uma medida científica de atenção. Não
há treinamento customizado, webcam em tempo real, backend, frontend ou banco de dados.

## Como executar

### Google Colab

Abra `deteccao_objetos_yolo.ipynb` e execute as células em ordem. Pelo painel lateral
de arquivos do Colab, envie o MP4 e mantenha o caminho único `/content/video.mp4`.
A primeira célula instala apenas Ultralytics ou MediaPipe que ainda não estejam
disponíveis. Antes do processamento completo, confira manualmente as amostras de
pessoa, telefone e direção da cabeça.

### Execução local opcional

Na raiz do workspace das entregas:

```bash
uv sync --all-packages
cd exercicios-03-yolo
uv run jupyter notebook deteccao_objetos_yolo.ipynb
```

Na célula de configurações, ajuste `INPUT_VIDEO_PATH` para o caminho do arquivo local,
por exemplo:

```python
INPUT_VIDEO_PATH = Path("video.mp4")
```

No Colab, mantenha `INPUT_VIDEO_PATH = Path("/content/video.mp4")`. Na primeira
utilização, as bibliotecas baixam automaticamente os pesos YOLO e o modelo Face
Landmarker. O vídeo anotado fica em `saidas_mvp/sessao_foco_anotada_opencv.mp4` quando
`CONVERT_OUTPUT_TO_H264 = False`.

Em GPU, o Ultralytics escolhe o dispositivo automaticamente. Em CPU, aumente
`PROCESS_EVERY_N_FRAMES` para reduzir o tempo de processamento sem alterar o relógio
baseado no FPS original.

## Regra da métrica

`focus_score = focused_time / total_valid_time * 100`

`total_valid_time` inclui tempo focado, distraído e ausente, mas exclui `UNKNOWN`.
`Distracted Time` já inclui `Absent Time`, que também é mostrado separadamente para
auditoria.

## Validação quantitativa do detector

Antes de processar o vídeo, o notebook mede o detector com o `coco128.yaml`, dataset
oficial da Ultralytics baixado automaticamente na primeira execução (7 MB). Ele contém
as duas classes que sustentam as regras do MVP, e a avaliação é restrita a elas com
`classes=[0, 67]`:

```python
metricas_coco = modelo.val(data="coco128.yaml", classes=[0, 67])
```

Resultado medido com `yolo26n.pt` em CPU:

| classe | precisão | recall | mAP50 | mAP50-95 | imagens | instâncias |
| --- | --- | --- | --- | --- | --- | --- |
| person | 0,862 | 0,661 | 0,791 | 0,549 | 61 | 254 |
| cell phone | 0,550 | 0,125 | 0,194 | 0,104 | 5 | 8 |

O mAP50 alto de `person` confirma que a detecção de presença é confiável. O valor baixo
de `cell phone` vem da amostra minúscula (8 instâncias, caixas pequenas) e é o que
justifica `PHONE_CONFIDENCE = 0.40`, mais permissivo que `PERSON_CONFIDENCE = 0.50`.

**Limitação:** o COCO128 reúne as 128 primeiras imagens do COCO *train2017*, conjunto
usado no pré-treino do `yolo26n.pt`. O modelo já viu essas imagens, então os números são
otimistas e valem como verificação de sanidade, não como avaliação em dados não vistos.
Remover esse viés exigiria o COCO val2017, cujo download passa de 19 GB e inviabiliza a
execução no Colab gratuito.

## Referências

- [Modo de predição do Ultralytics YOLO](https://docs.ultralytics.com/modes/predict/)
- [Dataset COCO128 da Ultralytics](https://docs.ultralytics.com/datasets/detect/coco128/)
- [MediaPipe Face Landmarker para Python](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [Artigo original: You Only Look Once](https://arxiv.org/abs/1506.02640)
