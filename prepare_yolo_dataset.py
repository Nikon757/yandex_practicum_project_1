"""
Подготовка разметки для YOLO: конвертация COCO JSON в формат YOLO TXT
и создание файла data.yaml.

Изображения общие для обеих моделей и не копируются:

    datasets/minecraft/
    ├── images/{train,valid,test}/                  <- изображения (одна копия)
    ├── annotations/{train,valid,test}/annotations.json   <- COCO для FCOS
    └── labels/{train,valid,test}/                  <- YOLO TXT, по одному .txt на изображение

Строка разметки YOLO: <class_id> <x_center> <y_center> <width> <height>,
координаты нормированы на размер изображения (от 0 до 1).

Аугментированные копии изображений с редкими классами (images/train_augmented,
labels/train_augmented) создаёт augment_dataset.py; в data.yaml обучающая выборка -
это две папки: train и train_augmented.
"""
import os
import random

import cv2
import yaml
import matplotlib.pyplot as plt

from constants import CLASSES, SPLITS, ANN_FILES, IMG_DIRS, LABELS_DIRS, DATA_YAML
from eda import load_coco, get_image_files


def coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    Переводит рамку из формата COCO [x_min, y_min, w, h] (пиксели)
    в формат YOLO [x_center, y_center, w, h] (доли от размера изображения).
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    return x_center, y_center, w / img_width, h / img_height


def convert_split(ann_file, out_labels_dir):
    """
    Конвертирует разметку одной выборки: пишет .txt-файл YOLO на каждое изображение.

    Args:
        ann_file (str): Путь к annotations.json выборки.
        out_labels_dir (str): Куда записать файлы разметки.

    Returns:
        tuple[int, int]: Количество изображений и количество рамок.
    """
    os.makedirs(out_labels_dir, exist_ok=True)

    coco = load_coco(ann_file)

    # id категории COCO -> индекс класса YOLO (0..16 в порядке CLASSES);
    # служебную категорию Roboflow 'minecraft-mobs' (id 0) пропускаем
    cat_id_to_class_id = {
        cat['id']: CLASSES.index(cat['name'])
        for cat in coco['categories'] if cat['name'] in CLASSES
    }

    # Группируем аннотации по изображениям
    anns_by_image = {img['id']: [] for img in coco['images']}
    for ann in coco['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    n_boxes = 0
    for img in coco['images']:
        lines = []
        for ann in anns_by_image[img['id']]:
            class_id = cat_id_to_class_id[ann['category_id']]
            x_center, y_center, w, h = coco_bbox_to_yolo(ann['bbox'], img['width'], img['height'])
            lines.append(f'{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}')
        n_boxes += len(lines)

        # Файл разметки называется как изображение, но с расширением .txt
        label_name = os.path.splitext(img['file_name'])[0] + '.txt'
        with open(os.path.join(out_labels_dir, label_name), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    return len(coco['images']), n_boxes


def write_data_yaml(data_yaml_path=DATA_YAML):
    """
    Создаёт data.yaml - «паспорт» датасета для YOLO.

    Пути относительные - от папки, в которой лежит сам data.yaml (Ultralytics берёт
    её за корень, если ключ path не задан). Обучающая выборка - две папки:
    исходные изображения и аугментированные копии с редкими классами.
    """
    data_config = {
        'train': ['images/train', 'images/train_augmented'],
        'val': 'images/valid',
        'test': 'images/test',
        'nc': len(CLASSES),
        'names': list(CLASSES),
    }
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data_config, f, sort_keys=False, allow_unicode=True)
    return data_config


def convert_annotations_to_yolo():
    """
    Конвертирует COCO-разметку всех выборок в формат YOLO TXT.
    Уже сконвертированные выборки пропускаются.
    """
    for split in SPLITS:
        n_expected = len(get_image_files(IMG_DIRS[split]))
        if os.path.isdir(LABELS_DIRS[split]) and len(os.listdir(LABELS_DIRS[split])) == n_expected:
            print(f'{split}: разметка YOLO уже подготовлена ({n_expected} файлов)')
            continue

        n_images, n_boxes = convert_split(ANN_FILES[split], LABELS_DIRS[split])
        print(f'{split}: {n_images} изображений, записано {n_boxes} рамок')


def prepare_yolo_dataset():
    """
    Готовит разметку YOLO для всех выборок и data.yaml.

    Returns:
        dict: Конфигурация датасета (содержимое data.yaml).
    """
    convert_annotations_to_yolo()
    data_config = write_data_yaml()
    print(f'Файл {DATA_YAML} записан: {data_config["nc"]} классов, train = {data_config["train"]}')
    return data_config


def verify_yolo_labels(save_path):
    """
    Проверка конвертации (как verify_and_visualize из урока о подготовке данных):
    читает data.yaml, берёт случайное изображение из обучающей выборки
    и рисует на нём рамки из .txt-файла.

    Args:
        save_path (str): Куда сохранить картинку.
    """
    with open(DATA_YAML, 'r', encoding='utf-8') as f:
        class_names = yaml.safe_load(f)['names']

    images_dir = IMG_DIRS['train']
    labels_dir = LABELS_DIRS['train']

    random.seed(42)
    image_name = random.choice(get_image_files(images_dir))
    label_name = os.path.splitext(image_name)[0] + '.txt'

    image = cv2.imread(os.path.join(images_dir, image_name))
    img_height, img_width, _ = image.shape
    thickness = max(2, img_width // 500)

    with open(os.path.join(labels_dir, label_name), 'r', encoding='utf-8') as f:
        for line in f.read().splitlines():
            parts = line.split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            # Денормализация: из долей обратно в пиксели
            box_w = int(width * img_width)
            box_h = int(height * img_height)
            x_min = int(x_center * img_width - box_w / 2)
            y_min = int(y_center * img_height - box_h / 2)

            cv2.rectangle(image, (x_min, y_min), (x_min + box_w, y_min + box_h), (0, 255, 0), thickness)
            cv2.putText(image, class_names[class_id], (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, thickness / 3, (0, 255, 0), thickness)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Проверка разметки YOLO: {image_name}')
    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    prepare_yolo_dataset()
