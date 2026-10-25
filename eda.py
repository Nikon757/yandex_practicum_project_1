"""
Исследовательский анализ датасета Minecraft в формате COCO.

Функции вызываются из ноутбука: знакомство с персонажами (этап 0)
и проверка данных перед обучением (этап 2). Скрипт можно запустить и отдельно -
тогда он выполнит все проверки и сохранит графики в artifacts/eda/.
"""
import os
import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from constants import (CLASSES, MOB_TYPES, MOB_TYPE_NAMES_RU, MOB_TYPE_COLORS,
                       SPLITS, IMG_DIRS, ANN_FILES, VIDEO_PATH, EDA_DIR)


def load_coco(ann_file):
    """Читает файл разметки в формате COCO и возвращает его содержимое как словарь."""
    with open(ann_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_image_files(directory):
    """Возвращает отсортированный список имён файлов изображений в папке."""
    return sorted(file_name for file_name in os.listdir(directory) if file_name.endswith('.jpg'))


def save_figure(save_path):
    """Сохраняет текущую фигуру matplotlib, создавая папку при необходимости."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')


# ----------------------------------------------------------------------------
# Этап 0. Знакомство с персонажами
# ----------------------------------------------------------------------------

def show_class_examples(ann_file, img_dir, save_path):
    """
    Показывает по одному примеру каждого класса из датасета.

    Для каждого класса подбирается «показательный» экземпляр: рамка не касается
    краёв кадра (объект не обрезан) и занимает около 10 % площади изображения -
    не крошечная и не крупный план. Фрагмент вырезается с небольшим отступом,
    чтобы был виден контекст, и подписывается именем класса и типом персонажа.

    Args:
        ann_file (str): Путь к annotations.json нужной выборки.
        img_dir (str): Папка с изображениями этой выборки.
        save_path (str): Куда сохранить картинку с сеткой примеров.
    """
    coco = load_coco(ann_file)

    # Справочники: id изображения -> описание изображения, id категории -> имя класса
    id_to_image = {img['id']: img for img in coco['images']}
    id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}

    # Для каждого класса запоминаем аннотацию, ближайшую к желаемому размеру
    target_fraction = 0.1
    best_ann = {}
    for ann in coco['annotations']:
        class_name = id_to_name[ann['category_id']]
        image_info = id_to_image[ann['image_id']]
        x, y, w, h = ann['bbox']

        # Пропускаем объекты, обрезанные краем кадра
        touches_border = (x <= 1 or y <= 1
                          or x + w >= image_info['width'] - 1
                          or y + h >= image_info['height'] - 1)
        if touches_border:
            continue

        fraction = (w * h) / (image_info['width'] * image_info['height'])
        score = abs(fraction - target_fraction)
        if class_name not in best_ann or score < best_ann[class_name]['score']:
            best_ann[class_name] = {'ann': ann, 'score': score}

    # Сетка 3 x 6 под 17 классов, лишнюю ячейку скрываем
    fig, axs = plt.subplots(3, 6, figsize=(18, 10))
    fig.suptitle('Персонажи датасета: по одному примеру на класс', fontsize=16)
    for ax in axs.flat:
        ax.axis('off')

    for ax, class_name in zip(axs.flat, CLASSES):
        ann = best_ann[class_name]['ann']

        image = cv2.imread(os.path.join(img_dir, id_to_image[ann['image_id']]['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # COCO хранит рамку как [x_min, y_min, width, height] в пикселях
        x, y, w, h = map(int, ann['bbox'])

        # Небольшой отступ вокруг рамки, чтобы был виден контекст
        pad = int(0.15 * max(w, h))
        crop = image[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]

        mob_type = MOB_TYPES[class_name]
        ax.imshow(crop)
        ax.set_title(f'{class_name}\n({MOB_TYPE_NAMES_RU[mob_type]})',
                     color=MOB_TYPE_COLORS[mob_type], fontsize=12)

    plt.tight_layout()
    save_figure(save_path)
    plt.show()


# ----------------------------------------------------------------------------
# Этап 2. Проверка структуры данных
# ----------------------------------------------------------------------------

def check_dataset_structure():
    """
    Проверяет, что все папки и файлы датасета на месте, и печатает отчёт.

    Ожидаемая структура:
        datasets/minecraft/
        ├─ images/{train,valid,test}/                    (изображения)
        ├─ annotations/{train,valid,test}/annotations.json
        └─ video.mp4
    """
    print('Структура датасета:')
    for split in SPLITS:
        n_images = len(get_image_files(IMG_DIRS[split]))
        print(f'  [OK] {IMG_DIRS[split]}/ - {n_images} изображений')
        status = 'OK' if os.path.exists(ANN_FILES[split]) else 'НЕТ'
        print(f'  [{status}] {ANN_FILES[split]}')
    status = 'OK' if os.path.exists(VIDEO_PATH) else 'НЕТ'
    print(f'  [{status}] {VIDEO_PATH}')


def check_coco_structure(ann_file):
    """
    Проверяет, что файл разметки соответствует формату COCO и нашему датасету:
    есть разделы images / annotations / categories, имена классов совпадают
    с CLASSES, каждая аннотация ссылается на существующее изображение и категорию,
    а рамка лежит внутри изображения.

    Args:
        ann_file (str): Путь к annotations.json.

    Returns:
        list[str]: Список найденных проблем (пустой список - разметка корректна).
    """
    coco = load_coco(ann_file)
    problems = []

    # 1. Обязательные разделы
    for section in ['images', 'annotations', 'categories']:
        if section not in coco:
            problems.append(f'нет раздела "{section}"')
    if problems:
        return problems

    # 2. Классы. Категория с id 0 - служебная 'minecraft-mobs' от Roboflow, объектов
    #    с ней в разметке нет; остальные имена должны совпадать с CLASSES по порядку.
    class_names = [cat['name'] for cat in coco['categories'] if cat['id'] != 0]
    if class_names != CLASSES:
        problems.append(f'список классов не совпадает с CLASSES: {class_names}')

    # 3. Аннотации: ссылки на изображение и категорию, положение рамки
    id_to_image = {img['id']: img for img in coco['images']}
    category_ids = {cat['id'] for cat in coco['categories']}
    for ann in coco['annotations']:
        if ann['image_id'] not in id_to_image:
            problems.append(f'аннотация {ann["id"]}: нет изображения с id {ann["image_id"]}')
            continue
        if ann['category_id'] not in category_ids:
            problems.append(f'аннотация {ann["id"]}: нет категории с id {ann["category_id"]}')
            continue

        img = id_to_image[ann['image_id']]
        x, y, w, h = ann['bbox']
        # Допуск в 1 пиксель на округление координат при экспорте
        if w <= 0 or h <= 0 or x < -1 or y < -1 or x + w > img['width'] + 1 or y + h > img['height'] + 1:
            problems.append(f'аннотация {ann["id"]}: рамка {ann["bbox"]} выходит за границы '
                            f'изображения {img["file_name"]} ({img["width"]}x{img["height"]})')

    print(f'{ann_file}: изображений {len(coco["images"])}, '
          f'аннотаций {len(coco["annotations"])}, классов {len(class_names)}')
    if problems:
        print(f'  Найдено проблем: {len(problems)}')
        for problem in problems[:10]:
            print(f'  - {problem}')
    else:
        print('  Структура JSON корректна')
    return problems


def check_images_match_annotations(ann_file, img_dir):
    """
    Сверяет изображения на диске с разметкой - через множества имён файлов,
    как при поиске пар «изображение - метка» в уроке об обучении YOLO.

    Args:
        ann_file (str): Путь к annotations.json.
        img_dir (str): Папка с изображениями.
    """
    coco = load_coco(ann_file)

    files_on_disk = set(get_image_files(img_dir))
    files_in_json = {img['file_name'] for img in coco['images']}
    annotated_image_ids = {ann['image_id'] for ann in coco['annotations']}

    files_without_records = files_on_disk - files_in_json
    records_without_files = files_in_json - files_on_disk
    images_without_annotations = [img['file_name'] for img in coco['images']
                                  if img['id'] not in annotated_image_ids]

    print(f'{img_dir}: файлов на диске {len(files_on_disk)}, записей в JSON {len(files_in_json)}, '
          f'рамок {len(coco["annotations"])}')
    if files_without_records:
        print(f'  Файлы без записи в разметке: {sorted(files_without_records)[:5]}')
    if records_without_files:
        print(f'  Записи разметки без файла на диске: {sorted(records_without_files)[:5]}')
    if images_without_annotations:
        print(f'  Изображения без единой рамки: {images_without_annotations[:5]}')
    if not (files_without_records or records_without_files or images_without_annotations):
        print('  Все изображения размечены, лишних файлов и записей нет')


# ----------------------------------------------------------------------------
# Этап 2. Анализ распределений
# ----------------------------------------------------------------------------

def analyze_class_distribution(save_path):
    """
    Считает количество объектов каждого класса в каждой выборке и строит диаграмму
    (как analyze_class_distribution из урока о подготовке данных).

    Args:
        save_path (str): Куда сохранить график.

    Returns:
        pd.DataFrame: Таблица «класс x выборка» с количеством объектов,
                      итогом и долей класса в обучающей выборке.
    """
    rows = []
    for split in SPLITS:
        coco = load_coco(ANN_FILES[split])
        id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
        class_counter = Counter(id_to_name[ann['category_id']] for ann in coco['annotations'])
        for class_name in CLASSES:
            rows.append({'split': split, 'class': class_name, 'count': class_counter[class_name]})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(14, 6))
    sns.set_theme(style='whitegrid')
    ax = sns.barplot(data=df, x='class', y='count', hue='split', palette='viridis')
    ax.set_title('Распределение классов по выборкам', fontsize=16)
    ax.set_xlabel('Класс', fontsize=12)
    ax.set_ylabel('Количество объектов', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_figure(save_path)
    plt.show()

    # Сводная таблица: классы в строках, выборки в столбцах
    table = df.pivot(index='class', columns='split', values='count')[SPLITS]
    table['total'] = table.sum(axis=1)
    table['train_share_%'] = (100 * table['train'] / table['train'].sum()).round(1)
    return table.sort_values('total', ascending=False)


def analyze_object_sizes(ann_file, img_scale, save_path):
    """
    Анализирует размеры изображений и объектов в выборке.

    Печатает разрешения изображений, строит гистограммы размера объектов после
    сжатия до img_scale и числа объектов на изображении. Размер объекта - длинная
    сторона рамки, пересчитанная на кадр с длинной стороной img_scale.

    Args:
        ann_file (str): Путь к annotations.json.
        img_scale (int): Размер длинной стороны, к которому будут сжиматься изображения.
        save_path (str): Куда сохранить график.
    """
    coco = load_coco(ann_file)
    id_to_image = {img['id']: img for img in coco['images']}

    # Разрешения изображений
    resolutions = Counter(f'{img["width"]}x{img["height"]}' for img in coco['images'])
    print('Разрешения изображений (ширина x высота):')
    for resolution, count in resolutions.most_common():
        print(f'  {resolution:>10}: {count}')

    # Размер каждого объекта после сжатия и число объектов на изображении
    sizes_at_scale = []
    for ann in coco['annotations']:
        img = id_to_image[ann['image_id']]
        _, _, w, h = ann['bbox']
        sizes_at_scale.append(max(w, h) / max(img['width'], img['height']) * img_scale)
    sizes_at_scale = np.array(sizes_at_scale)
    objects_per_image = np.array(list(Counter(ann['image_id'] for ann in coco['annotations']).values()))

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].hist(sizes_at_scale, bins=50, color='steelblue')
    axs[0].axvline(32, color='red', linestyle='--', label='32 px')
    axs[0].set_title(f'Размер объекта после сжатия до {img_scale} px по длинной стороне')
    axs[0].set_xlabel('Длинная сторона рамки, px')
    axs[0].set_ylabel('Количество объектов')
    axs[0].legend()

    axs[1].hist(objects_per_image, bins=np.arange(1, objects_per_image.max() + 2) - 0.5,
                color='steelblue')
    axs[1].set_title('Количество объектов на изображении')
    axs[1].set_xlabel('Объектов на изображении')
    axs[1].set_ylabel('Количество изображений')
    plt.tight_layout()
    save_figure(save_path)
    plt.show()

    print(f'\nОбъектов: {len(sizes_at_scale)}, медианный размер после сжатия: '
          f'{np.median(sizes_at_scale):.1f} px, мелких (< 32 px): {100 * np.mean(sizes_at_scale < 32):.1f}%')
    print(f'Объектов на изображении: в среднем {objects_per_image.mean():.2f}, '
          f'максимум {objects_per_image.max()}')


# ----------------------------------------------------------------------------
# Этап 2. Визуализация разметки
# ----------------------------------------------------------------------------

def visualize_sample(ann_file, img_dir, file_name, save_path):
    """
    Показывает одно изображение с истинными рамками и подписями классов
    (рамки рисуются так же, как в уроке о подготовке данных).

    Args:
        ann_file (str): Путь к annotations.json.
        img_dir (str): Папка с изображениями.
        file_name (str): Имя файла изображения из разметки.
        save_path (str): Куда сохранить картинку.
    """
    coco = load_coco(ann_file)
    id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
    image_id = next(img['id'] for img in coco['images'] if img['file_name'] == file_name)
    anns = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]

    image = cv2.imread(os.path.join(img_dir, file_name))
    for ann in anns:
        # COCO хранит рамку как [x_min, y_min, width, height] в пикселях
        x_min, y_min, w, h = map(int, ann['bbox'])
        class_name = id_to_name[ann['category_id']]
        cv2.rectangle(image, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'{file_name}\nобъектов: {len(anns)}')
    plt.axis('off')
    save_figure(save_path)
    plt.show()


if __name__ == '__main__':
    # Полный прогон всех проверок и графиков (как в ноутбуке), графики - в artifacts/eda/
    check_dataset_structure()
    for split in SPLITS:
        check_coco_structure(ANN_FILES[split])
        check_images_match_annotations(ANN_FILES[split], IMG_DIRS[split])

    show_class_examples(ANN_FILES['train'], IMG_DIRS['train'], os.path.join(EDA_DIR, 'mob_classes.png'))
    print(analyze_class_distribution(os.path.join(EDA_DIR, 'class_distribution.png')))
    analyze_object_sizes(ANN_FILES['train'], 512, os.path.join(EDA_DIR, 'object_sizes.png'))
    visualize_sample(ANN_FILES['test'], IMG_DIRS['test'],
                     'grass_desert_-_cow_3_1434_jpg.rf.110c73a314ed050344613c069fbc1328.jpg',
                     os.path.join(EDA_DIR, 'test_sample_gt.png'))
