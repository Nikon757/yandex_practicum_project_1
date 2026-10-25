"""
Офлайн-аугментация изображений с редкими классами перед обучением
(по уроку о подготовке данных: Albumentations сама пересчитывает координаты рамок).

Редких классов в обучающей выборке мало (wolf - 4 рамки, enderman - 22, ...),
поэтому «размножаем» изображения с ними: для каждого такого изображения создаём
несколько аугментированных копий. Какие классы редкие и сколько копий делать,
задано явно в constants.RARE_CLASS_REPEATS.

Копии складываются рядом с исходными данными (datasets/minecraft/):
    images/train + labels/train  ->  images/train_augmented + labels/train_augmented
Для FCOS та же разметка дополнительно записывается в формате COCO:
    annotations/train_augmented/annotations.json

Запуск из консоли: python augment_dataset.py
"""
import os
import json
import random
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A

from constants import (CLASSES, RARE_CLASS_REPEATS, ANN_FILES, IMG_DIRS, LABELS_DIRS,
                       AUG_IMG_DIR, AUG_LABELS_DIR, AUG_ANN_FILE)
from eda import load_coco, get_image_files

# Исходные изображения и разметка обучающей выборки в формате YOLO
TRAIN_IMG_DIR = IMG_DIRS['train']
TRAIN_LABELS_DIR = LABELS_DIRS['train']

# Пайплайн аугментаций из урока: геометрические, цветовые и шум.
# Поворот отключён (rotate_limit=0): скриншоты Minecraft не бывают наклонёнными,
# а аугментация должна имитировать реальные условия.
transform = A.Compose([
    # Геометрические аугментации
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=0, p=0.7),

    # Цветовые аугментации
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),

    A.GaussNoise(p=0.2)
],
    bbox_params=A.BboxParams(
        format='yolo',                 # рамки в формате [x_center, y_center, w, h], нормированные
        label_fields=['category_ids']  # просим менять и id классов вместе с рамками
    )
)

# Аугментация «с сокрытием» (учит модель работать с окклюзиями). Она не двигает
# пиксели, координаты рамок не меняются, а в albumentations 1.3.1 CoarseDropout
# не принимает рамки - поэтому применяем её отдельным шагом только к изображению.
occlusion = A.Compose([
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)
])


def read_yolo_labels(label_path):
    """
    Читает файл разметки YOLO: возвращает список рамок и список id классов.

    Roboflow записывает часть рамок на пиксель-другой за границей кадра,
    а Albumentations требует координаты строго в [0, 1] - обрезаем рамки по кадру.
    """
    bboxes = []
    category_ids = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f.read().splitlines():
            parts = line.split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])

            x_min = max(0.0, x_center - width / 2)
            y_min = max(0.0, y_center - height / 2)
            x_max = min(1.0, x_center + width / 2)
            y_max = min(1.0, y_center + height / 2)

            bboxes.append([(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min])
            category_ids.append(class_id)
    return bboxes, category_ids


def write_yolo_labels(label_path, bboxes, category_ids):
    """Записывает рамки в файл разметки YOLO."""
    lines = [
        f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'
        for (x_center, y_center, width, height), class_id in zip(bboxes, category_ids)
    ]
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def visualize(image, bboxes, category_ids, category_id_to_name):
    """Рисует рамки и подписи на изображении (функция из урока), возвращает копию."""
    img_viz = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]

        # Денормализуем координаты из формата YOLO в пиксельные координаты
        img_h, img_w, _ = image.shape
        x_center, y_center, width, height = bbox
        x_min = int((x_center - width / 2) * img_w)
        y_min = int((y_center - height / 2) * img_h)
        x_max = int((x_center + width / 2) * img_w)
        y_max = int((y_center + height / 2) * img_h)

        # Рисуем рамку и подпись
        cv2.rectangle(img_viz, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
        cv2.putText(img_viz, class_name, (x_min, y_min - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(0, 255, 0), thickness=2)
    return img_viz


def augment_rare_classes():
    """
    Создаёт аугментированные копии изображений, на которых есть редкие классы.

    Для изображения делается (N - 1) копий, где N - наибольшее число показов среди
    его классов из RARE_CLASS_REPEATS (вместе с оригиналом класс увидят N раз за эпоху).
    Seed фиксирован, поэтому копии воспроизводимы.

    Returns:
        Counter: Количество аугментированных рамок по классам.
    """
    os.makedirs(AUG_IMG_DIR, exist_ok=True)
    os.makedirs(AUG_LABELS_DIR, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    augmented_boxes = Counter()
    n_images = 0
    for image_name in get_image_files(TRAIN_IMG_DIR):
        stem = os.path.splitext(image_name)[0]
        bboxes, category_ids = read_yolo_labels(os.path.join(TRAIN_LABELS_DIR, stem + '.txt'))

        # Сколько копий нужно изображению - по самому редкому из его классов
        n_copies = max(RARE_CLASS_REPEATS.get(CLASSES[class_id], 1) for class_id in category_ids) - 1
        if n_copies == 0:
            continue

        # Albumentations работает с RGB, cv2 читает BGR
        image = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_IMG_DIR, image_name)), cv2.COLOR_BGR2RGB)

        for k in range(1, n_copies + 1):
            # Передаём в пайплайн и изображение, и его разметку
            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            aug_image = occlusion(image=transformed['image'])['image']

            cv2.imwrite(os.path.join(AUG_IMG_DIR, f'{stem}_aug{k}.jpg'),
                        cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            write_yolo_labels(os.path.join(AUG_LABELS_DIR, f'{stem}_aug{k}.txt'),
                              transformed['bboxes'], transformed['category_ids'])

            n_images += 1
            augmented_boxes.update(CLASSES[class_id] for class_id in transformed['category_ids'])

    print(f'Создано {n_images} аугментированных изображений в {AUG_IMG_DIR}')
    return augmented_boxes


def yolo_to_coco():
    """
    Собирает COCO-разметку аугментированных копий для FCOS (AUG_ANN_FILE).

    Раздел categories берётся из исходной разметки, category_id - как в ней:
    индекс класса + 1 (id 0 занят служебной категорией Roboflow).
    """
    categories = load_coco(ANN_FILES['train'])['categories']

    images = []
    annotations = []
    for image_id, image_name in enumerate(get_image_files(AUG_IMG_DIR)):
        img_h, img_w, _ = cv2.imread(os.path.join(AUG_IMG_DIR, image_name)).shape
        images.append({'id': image_id, 'file_name': image_name, 'width': img_w, 'height': img_h})

        stem = os.path.splitext(image_name)[0]
        bboxes, category_ids = read_yolo_labels(os.path.join(AUG_LABELS_DIR, stem + '.txt'))
        for (x_center, y_center, width, height), class_id in zip(bboxes, category_ids):
            w, h = width * img_w, height * img_h
            x, y = x_center * img_w - w / 2, y_center * img_h - h / 2
            annotations.append({
                'id': len(annotations),
                'image_id': image_id,
                'category_id': class_id + 1,
                'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                'area': round(w * h, 2),
                'segmentation': [],
                'iscrowd': 0,
            })

    os.makedirs(os.path.dirname(AUG_ANN_FILE), exist_ok=True)
    with open(AUG_ANN_FILE, 'w', encoding='utf-8') as f:
        json.dump({'categories': categories, 'images': images, 'annotations': annotations}, f)
    print(f'COCO-разметка для FCOS: {AUG_ANN_FILE} ({len(images)} изображений, {len(annotations)} рамок)')


def augmentation_summary(augmented_boxes):
    """Таблица: рамок каждого класса в исходной обучающей выборке, добавлено и итого."""
    coco = load_coco(ANN_FILES['train'])
    id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    original_boxes = Counter(id_to_name[ann['category_id']] for ann in coco['annotations'])

    table = pd.DataFrame({
        'copies_per_image': [RARE_CLASS_REPEATS.get(c, 1) - 1 for c in CLASSES],
        'boxes_original': [original_boxes[c] for c in CLASSES],
        'boxes_augmented': [augmented_boxes[c] for c in CLASSES],
    }, index=CLASSES)
    table['boxes_total'] = table['boxes_original'] + table['boxes_augmented']
    return table.sort_values('boxes_total')


def plot_augmentation_effect(table, save_path):
    """
    Показывает, как аугментация изменила распределение классов в обучающей выборке:
    по два столбца на класс - сколько рамок было и сколько стало.

    Args:
        table (pd.DataFrame): Таблица из augmentation_summary.
        save_path (str): Куда сохранить график.
    """
    # Длинный формат: на каждый класс по две строки - «было» и «стало»
    df = table.reset_index().rename(columns={'index': 'class'})
    df = df.melt(id_vars='class', value_vars=['boxes_original', 'boxes_total'],
                 var_name='stage', value_name='boxes')
    df['stage'] = df['stage'].map({'boxes_original': 'было', 'boxes_total': 'стало'})

    plt.figure(figsize=(14, 6))
    sns.set_theme(style='whitegrid')
    ax = sns.barplot(data=df, x='class', y='boxes', hue='stage', palette='viridis')
    ax.set_title('Рамки в обучающей выборке до и после аугментации редких классов', fontsize=16)
    ax.set_xlabel('Класс', fontsize=12)
    ax.set_ylabel('Количество рамок', fontsize=12)
    ax.legend(title='')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def show_augmentation_example(image_name):
    """Показывает исходное изображение и три его аугментированные копии (проверка перед обучением)."""
    category_id_to_name = dict(enumerate(CLASSES))
    stem = os.path.splitext(image_name)[0]

    image = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_IMG_DIR, image_name)), cv2.COLOR_BGR2RGB)
    bboxes, category_ids = read_yolo_labels(os.path.join(TRAIN_LABELS_DIR, stem + '.txt'))
    panels = [('Исходное изображение', visualize(image, bboxes, category_ids, category_id_to_name))]

    for k in range(1, 4):
        aug_image = cv2.cvtColor(cv2.imread(os.path.join(AUG_IMG_DIR, f'{stem}_aug{k}.jpg')), cv2.COLOR_BGR2RGB)
        aug_bboxes, aug_category_ids = read_yolo_labels(os.path.join(AUG_LABELS_DIR, f'{stem}_aug{k}.txt'))
        panels.append((f'Копия {k}', visualize(aug_image, aug_bboxes, aug_category_ids, category_id_to_name)))

    fig, axs = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5))
    for ax, (title, panel) in zip(axs, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    augmented_boxes = augment_rare_classes()
    yolo_to_coco()
    print(augmentation_summary(augmented_boxes))
