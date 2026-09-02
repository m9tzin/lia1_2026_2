# MVP de Monitoramento Visual de Foco

**Aluno:** Matheus Sousa Marinho · **Matrícula:** 202206132 · **Disciplina:** LIA 1 (2026/2)

Esta entrega evolui o exercício de YOLO26 para um MVP executado inteiramente em um
notebook Google Colab. Ele recebe um vídeo gravado de uma sessão de estudo e usa
heurísticas visuais para estimar períodos focados, distraídos, ausentes e sem evidência
suficiente (`UNKNOWN`). O notebook:

- reaproveita o modelo pré-treinado `yolo26n.pt` para `person` e `cell phone`;
- usa MediaPipe Face Landmarker para orientação aproximada da cabeça;
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

## Referências

- [Modo de predição do Ultralytics YOLO](https://docs.ultralytics.com/modes/predict/)
- [MediaPipe Face Landmarker para Python](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [Artigo original: You Only Look Once](https://arxiv.org/abs/1506.02640)
