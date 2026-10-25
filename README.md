# Детекция персонажей Minecraft: FCOS против YOLOv8s

Учебный проект курса «Компьютерное зрение» Яндекс Практикума (спринт «Детекция объектов»).

## Задача

Дообучить два детектора объектов — anchor-free модель **FCOS** (MMDetection) и **YOLOv8s** (Ultralytics) —
распознавать 17 классов персонажей (мобов) Minecraft, прогнать их на тестовых изображениях и видео
и сравнить по качеству (mAP, mAP@0.5) и скорости инференса (FPS).

## Данные

Датасет в формате COCO (экспорт Roboflow): 2307 изображений в `train`, 422 в `valid`, 155 в `test`,
17 классов: `bee, chicken, cow, creeper, enderman, fox, frog, ghast, goat, llama, pig, sheep, skeleton,
spider, turtle, wolf, zombie`. Разрешения кадров — от 640×360 до 2560×1440.

Особенности, найденные при анализе данных:

- сильный дисбаланс классов в обучающей выборке (`pig` — 884 рамки, `wolf` — 4);
- в `valid` нет семи классов, в `test` — восьми, поэтому метрики на тесте считаются по 9 классам;
- 27 % объектов после сжатия до 640 px оказываются мельче 32 px (при 512 px было бы 37 %).

Дисбаланс компенсирован офлайн-аугментацией: изображения с редкими классами размножены пайплайном
Albumentations (`augment_dataset.py`), обучающая выборка выросла с 2307 до 3177 изображений.

## Результаты

Метрики на тестовой выборке (`artifacts/metrics/metrics_comparison.csv`):

| Модель | mAP@0.5:0.95 | mAP@0.5 | FPS |
|---|---|---|---|
| FCOS (ResNet-50 + FPN, 640 px, 16 эпох) | 0.500 | 0.773 | 41 |
| YOLOv8s (640 px, 30 эпох) | **0.558** | **0.798** | **110** |

Обе модели обучены в одном разрешении 640 px на RTX 4060 Ti: FCOS — 26 минут, YOLOv8s — 17 минут.

Выводы: YOLOv8s точнее (+0.058 mAP@0.5:0.95, лучше на семи классах из девяти) и в 2,7 раза быстрее —
единственная из двух моделей, пригодная для видео 60 кадров в секунду в реальном времени. FCOS выигрывает
только на двух ночных классах с единицами рамок в тесте (`skeleton`, `zombie`). Слабое место обеих моделей —
мелкие и дальние объекты: при 640 px 27 % объектов датасета меньше 32 пикселей.

Подробный отчёт с графиками и примерами детекций — `artifacts/report.pdf`.

## Структура репозитория

```
mmdetection/                          # корень проекта (клон MMDetection)
├── notebook.ipynb                    # все этапы работы: EDA, настройка, обучение, инференс, метрики, выводы
├── constants.py                      # классы, типы персонажей, пути к данным и артефактам
├── eda.py                            # проверки датасета и исследовательский анализ
├── prepare_yolo_dataset.py           # конвертация COCO -> YOLO TXT, data.yaml
├── augment_dataset.py                # аугментация изображений с редкими классами (Albumentations)
├── detection_models.py               # единый инференс FCOS/YOLO, draw_boxes, run_and_visualize, measure_fps
├── train_fcos.py                     # обучение FCOS (Runner), графики по логам
├── train_yolo.py                     # обучение YOLOv8s, графики по results.csv
├── process_video.py                  # покадровый инференс на видео
├── evaluate.py                       # mAP / mAP@0.5 на тесте, таблица сравнения
├── report.py                         # PDF-отчёт (класс PDFReport на fpdf2)
├── configs/fcos/fcos_minecraft.py    # конфиг дообучения FCOS
├── checkpoints/                      # предобученные веса (fcos_r50_caffe_fpn_gn-head_1x_coco, yolov8s.pt)
├── ttf/                              # шрифты DejaVu для PDF
├── datasets/minecraft/               # данные (в репозиторий не входят)
│   ├── images/{train,valid,test,train_augmented}/       # изображения - одна копия для обеих моделей
│   ├── annotations/{train,valid,test,train_augmented}/  # COCO JSON для FCOS
│   ├── labels/{train,valid,test,train_augmented}/       # YOLO TXT для YOLO
│   ├── data.yaml
│   └── video.mp4
└── artifacts/
    ├── eda/                          # графики анализа данных
    ├── fcos/                         # логи, чекпоинты и метрики FCOS
    ├── yolo/                         # логи, веса и results.csv YOLO
    ├── inference/                    # детекции на тестовых изображениях (fcos/, yolo/, comparison/)
    ├── videos/                       # fcos_inference.mp4, yolo_inference.mp4 (не в репозитории)
    ├── metrics/                      # metrics_comparison.csv и график
    └── report.pdf                    # итоговый отчёт
```

## Как запустить

В репозитории лежат только код, ноутбук и результаты работы. MMDetection, датасет, предобученные
и обученные веса не входят в репозиторий — их нужно получить самостоятельно:

```bash
# 1. Склонировать MMDetection - он и есть корень проекта
git clone https://github.com/open-mmlab/mmdetection.git
# 2. Скопировать в mmdetection/ содержимое этого репозитория
#    (notebook.ipynb, *.py, requirements.txt, configs/fcos/fcos_minecraft.py, artifacts/)
```

Окружение — комбинация версий из курса (Python 3.10):

```bash
python -m pip install pip==21.2.3
python -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -U openmim
python -m pip install "mmengine==0.10.7"
python -m mim install "mmcv==2.1.0"
python -m pip install -v -e . --no-build-isolation     # из папки mmdetection
python -m pip install -r requirements.txt
```

[Датасет](https://disk.yandex.ru/d/ezpvkg_cdDJnNA) распаковать в `datasets/minecraft/`: изображения — в
`images/<выборка>/`, разметку — в `annotations/<выборка>/annotations.json`; [видео](https://code.s3.yandex.net/deep-learning-cv/video.mp4) —
в `datasets/minecraft/video.mp4`. Разметку YOLO (`labels/`) и аугментированные копии создают скрипты
`prepare_yolo_dataset.py` и `augment_dataset.py`.

Предобученные веса: чекпоинт FCOS
[`fcos_r50_caffe_fpn_gn-head_1x_coco`](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth)
положить в `checkpoints/`, `yolov8s.pt` скачается сам при первом запуске. Шрифты DejaVu для PDF-отчёта —
[из материалов курса](https://code.s3.yandex.net/deep-learning/fonts_cv.zip), распаковать в `ttf/`
(без них отчёт соберётся стандартным шрифтом, но без кириллицы).

Обученные веса и видео с детекциями в репозиторий не загружены (лучший чекпоинт FCOS — 130 МБ,
каждое видео — около 60 МБ). Веса получаются ячейками обучения в ноутбуке (FCOS — 26 минут,
YOLOv8s — 17 минут на RTX 4060 Ti), видео — ячейками этапа 5 (около двух минут); отдельные кадры
из этих видео сохранены в ноутбуке и в `artifacts/report.pdf`.

Затем из папки `mmdetection/` запустить `python -m notebook` и выполнить `notebook.ipynb` сверху вниз.
Каждый этап можно запустить и отдельно из консоли, например:

```bash
python augment_dataset.py
python train_fcos.py
python train_yolo.py --epochs 30
python evaluate.py --fcos artifacts/fcos/best_coco_bbox_mAP_epoch_12.pth --yolo artifacts/yolo/yolov8s_minecraft/weights/best.pt
python process_video.py --model yolo --weights artifacts/yolo/yolov8s_minecraft/weights/best.pt
python report.py
```
