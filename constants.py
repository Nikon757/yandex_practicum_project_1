"""
Общие константы проекта: классы датасета, типы персонажей, пути к данным и артефактам.

Все пути относительные и считаются от корня проекта (папки mmdetection/),
из которой запускаются ноутбук и скрипты.
"""
import os

import numpy as np


# Классы датасета в том порядке, в котором они идут в annotations.json (id 1..17).
# Категорию 'minecraft-mobs' (id 0) Roboflow добавляет как служебную supercategory,
# объектов с ней в разметке нет, поэтому в список классов она не входит.
CLASSES = [
    'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
    'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'
]

# Тип персонажа по официальной вики Minecraft:
#   friendly - дружелюбный (никогда не нападает на игрока),
#   neutral  - нейтральный (нападает только в ответ или при определённых условиях),
#   hostile  - враждебный (нападает первым).
MOB_TYPES = {
    'bee': 'neutral',
    'chicken': 'friendly',
    'cow': 'friendly',
    'creeper': 'hostile',
    'enderman': 'neutral',
    'fox': 'friendly',
    'frog': 'friendly',
    'ghast': 'hostile',
    'goat': 'neutral',
    'llama': 'neutral',
    'pig': 'friendly',
    'sheep': 'friendly',
    'skeleton': 'hostile',
    'spider': 'neutral',
    'turtle': 'friendly',
    'wolf': 'neutral',
    'zombie': 'hostile',
}

# Русские названия типов и цвета для подписей на графиках
MOB_TYPE_NAMES_RU = {
    'friendly': 'дружелюбный',
    'neutral': 'нейтральный',
    'hostile': 'враждебный',
}
MOB_TYPE_COLORS = {
    'friendly': 'green',
    'neutral': 'darkorange',
    'hostile': 'red',
}

# Редкие классы обучающей выборки (по таблице распределения на этапе 2) и сколько раз
# в сумме показывать изображения с ними: оригинал + (N - 1) аугментированных копий,
# которые создаёт augment_dataset.py. Подобрано так, чтобы у каждого класса было
# не меньше ~170 рамок; wolf снят всего на 2 изображениях, поэтому больше 8 показов
# не даём - это уже заучивание двух картинок.
RARE_CLASS_REPEATS = {
    'wolf': 8,      # 4 рамки
    'enderman': 4,  # 22 рамки
    'bee': 2,       # 86 рамок
    'sheep': 2,     # 94 рамки
    'goat': 2,      # 102 рамки
    'fox': 2,       # 111 рамок
    'frog': 2,      # 111 рамок
    'ghast': 2,     # 127 рамок
    'llama': 2,     # 135 рамок
    'turtle': 2,    # 149 рамок
}


# Фиксированная палитра цветов для отрисовки рамок: один цвет на класс
# (как в уроке о сравнении моделей)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

# Данные. Изображения хранятся в одном экземпляре в images/<выборка>/ и используются
# обеими моделями; разметка для FCOS - COCO JSON в annotations/<выборка>/,
# для YOLO - текстовые файлы в labels/<выборка>/ (Ultralytics находит их,
# заменяя images/ на labels/ в пути к изображению).
DATA_ROOT = 'datasets/minecraft'
SPLITS = ['train', 'valid', 'test']
IMG_DIRS = {split: os.path.join(DATA_ROOT, 'images', split) for split in SPLITS}
ANN_FILES = {
    split: os.path.join(DATA_ROOT, 'annotations', split, 'annotations.json')
    for split in SPLITS
}
LABELS_DIRS = {split: os.path.join(DATA_ROOT, 'labels', split) for split in SPLITS}
VIDEO_PATH = os.path.join(DATA_ROOT, 'video.mp4')

# Тестовые изображения для примеров детекций (этапы 5-7): разные классы и разрешения
TEST_EXAMPLES = [
    'grass_desert_-_cow_3_1434_jpg.rf.110c73a314ed050344613c069fbc1328.jpg',   # cow x3, pig x3
    '2023-04-03_19_11_49_png.rf.28979ff2597860599925ba4f6ad99aec.jpg',        # zombie x2, skeleton
    'creeper-60-_png.rf.12403f46c516d763ca5c42fae7e5ac90.jpg',                # creeper
    '161_png_jpg.rf.21c98248d79b0987391e2f8ed834e367.jpg',                    # chicken x3, sheep
    '2023-04-03_20_31_22_png.rf.531dd945e6deeefaf5d3e681c58a0645.jpg',        # enderman
]

# Описание датасета для Ultralytics (создаёт prepare_yolo_dataset.py)
DATA_YAML = os.path.join(DATA_ROOT, 'data.yaml')
# Аугментированные копии изображений с редкими классами (создаёт augment_dataset.py):
# та же структура - изображения, YOLO-разметка и COCO-разметка для FCOS
AUG_IMG_DIR = os.path.join(DATA_ROOT, 'images', 'train_augmented')
AUG_LABELS_DIR = os.path.join(DATA_ROOT, 'labels', 'train_augmented')
AUG_ANN_FILE = os.path.join(DATA_ROOT, 'annotations', 'train_augmented', 'annotations.json')

# Конфиг и предобученные веса
FCOS_CONFIG = 'configs/fcos/fcos_minecraft.py'
FCOS_PRETRAINED = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'
YOLO_PRETRAINED = 'checkpoints/yolov8s.pt'

# Артефакты проекта
ARTIFACTS_DIR = 'artifacts'
EDA_DIR = os.path.join(ARTIFACTS_DIR, 'eda')
FCOS_WORK_DIR = os.path.join(ARTIFACTS_DIR, 'fcos')        # логи, чекпоинты, метрики FCOS
YOLO_WORK_DIR = os.path.join(ARTIFACTS_DIR, 'yolo')        # логи, веса, results.csv YOLO
YOLO_EXPERIMENT = 'yolov8s_minecraft'                       # имя эксперимента (подпапка в YOLO_WORK_DIR)
INFERENCE_DIR = os.path.join(ARTIFACTS_DIR, 'inference')
FCOS_INFERENCE_DIR = os.path.join(INFERENCE_DIR, 'fcos')
YOLO_INFERENCE_DIR = os.path.join(INFERENCE_DIR, 'yolo')
YOLO_VAL_DIR = os.path.join(INFERENCE_DIR, 'yolo_val')      # проверка предобученной YOLO
VIDEOS_DIR = os.path.join(ARTIFACTS_DIR, 'videos')
METRICS_DIR = os.path.join(ARTIFACTS_DIR, 'metrics')
