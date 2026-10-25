"""
Итоговый PDF-отчёт artifacts/report.pdf - по уроку об анализе трафика:
класс PDFReport на fpdf2 со шрифтами DejaVu из папки ttf/ и сборка отчёта
из результатов этапов 4-6 (таблица метрик, графики, примеры детекций).

Запуск из консоли: python report.py
"""
import os
import datetime
from collections import Counter

import cv2
import yaml
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from mmengine.config import Config

from constants import (CLASSES, RARE_CLASS_REPEATS, SPLITS, ANN_FILES, FCOS_CONFIG, FCOS_WORK_DIR,
                       YOLO_WORK_DIR, YOLO_EXPERIMENT, INFERENCE_DIR, FCOS_INFERENCE_DIR,
                       YOLO_INFERENCE_DIR, METRICS_DIR, ARTIFACTS_DIR)
from eda import load_coco

REPORT_PATH = os.path.join(ARTIFACTS_DIR, 'report.pdf')
COMPARISON_IMAGES_DIR = os.path.join(INFERENCE_DIR, 'comparison')


class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Пробуем использовать красивые шрифты, если они доступны (их мы скачивали в нашу папку ttf/)
        regular_font_path = os.path.join('ttf', 'DejaVuSans.ttf')
        bold_font_path = os.path.join('ttf', 'DejaVuSans-Bold.ttf')

        if os.path.exists(regular_font_path) and os.path.exists(bold_font_path):
            self.add_font('DejaVu', '', regular_font_path)
            self.add_font('DejaVu', 'B', bold_font_path)

            self.font_family = 'DejaVu'
        else:
            # Если шрифты не найдены, используем стандартный
            self.font_family = 'Arial'

    def header(self):
        """Создаёт шапку для каждой страницы"""
        self.set_font(self.font_family, 'B', 15)
        self.cell(0, 10, 'Детекция персонажей Minecraft: FCOS против YOLOv8s', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font(self.font_family, '', 8)
        self.cell(0, 5, f'Дата генерации: {datetime.date.today().strftime("%d.%m.%Y")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        """Добавляет номера страниц в подвале"""
        self.set_y(-15)
        self.set_font(self.font_family, 'B', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

    def chapter_title(self, title):
        """Создаёт заголовок раздела"""
        self.set_font(self.font_family, 'B', 12)
        self.cell(0, 10, title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)

    def chapter_body(self, body):
        """Добавляет основной текст"""
        self.set_font(self.font_family, '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_image_section(self, title, image_path, stats_text, image_width=100):
        """Добавляет секцию с изображением и статистикой"""
        self.add_page()
        self.chapter_title(title)

        # Центрируем изображение на странице
        page_width = self.w - 2 * self.l_margin
        x_position = (page_width - image_width) / 2 + self.l_margin
        self.image(image_path, x=x_position, y=None, w=image_width)
        self.ln(5)
        self.set_font(self.font_family, '', 10)
        self.multi_cell(0, 5, stats_text)


def make_comparison_image(image_name):
    """
    Склеивает результаты FCOS и YOLO на одном тестовом изображении в одну картинку
    (слева FCOS, справа YOLOv8s) для вставки в отчёт.
    """
    os.makedirs(COMPARISON_IMAGES_DIR, exist_ok=True)
    fcos_image = cv2.imread(os.path.join(FCOS_INFERENCE_DIR, image_name))
    yolo_image = cv2.imread(os.path.join(YOLO_INFERENCE_DIR, image_name))

    save_path = os.path.join(COMPARISON_IMAGES_DIR, image_name)
    cv2.imwrite(save_path, cv2.hconcat([fcos_image, yolo_image]))
    return save_path


def build_report(quality_conclusion, speed_conclusion, example_images, output_path=REPORT_PATH):
    """
    Собирает PDF-отчёт: описание проекта, метрики качества и скорости с выводами,
    графики сравнения и обучения, примеры детекций обеих моделей.

    Args:
        quality_conclusion (str): Вывод о моделях по метрикам качества.
        speed_conclusion (str): Вывод о моделях по скорости инференса.
        example_images (list[str]): Имена тестовых изображений для примеров детекций.
        output_path (str): Куда сохранить PDF.
    """
    comparison = pd.read_csv(os.path.join(METRICS_DIR, 'metrics_comparison.csv'), index_col='model')
    fcos, yolo = comparison.loc['FCOS'], comparison.loc['YOLOv8s']

    # Параметры обучения берём из конфига FCOS и из args.yaml эксперимента YOLO
    fcos_cfg = Config.fromfile(FCOS_CONFIG)
    with open(os.path.join(YOLO_WORK_DIR, YOLO_EXPERIMENT, 'args.yaml'), 'r', encoding='utf-8') as f:
        yolo_args = yaml.safe_load(f)

    pdf = PDFReport()
    pdf.add_page()

    # 1. Описание проекта
    pdf.chapter_title('1. Описание проекта')
    split_sizes = {split: load_coco(ANN_FILES[split]) for split in SPLITS}
    dataset_text = ', '.join(
        f'{split}: {len(coco["images"])} изображений / {len(coco["annotations"])} рамок'
        for split, coco in split_sizes.items()
    )
    rare_text = ', '.join(f'{name} x{repeats}' for name, repeats in RARE_CLASS_REPEATS.items())
    description_text = (
        f'Задача: дообучить два детектора - anchor-free FCOS (MMDetection) и YOLOv8s (Ultralytics) - '
        f'распознавать {len(CLASSES)} классов персонажей Minecraft и сравнить их по качеству '
        f'(mAP, mAP@0.5) и скорости (FPS).\n\n'
        f'Данные: датасет в формате COCO ({dataset_text}). Классы: {", ".join(CLASSES)}.\n'
        f'Обучающая выборка сильно несбалансирована (pig - 884 рамки, wolf - 4), поэтому изображения '
        f'с редкими классами были размножены аугментациями Albumentations перед обучением '
        f'(число показов класса: {rare_text}).\n\n'
        f'FCOS: ResNet-50 + FPN, дообучение с весов COCO, изображения {fcos_cfg.img_scale[0]} px, '
        f'{fcos_cfg.train_cfg.max_epochs} эпох, батч {fcos_cfg.train_dataloader.batch_size}, '
        f'fp16, аугментации Resize + RandomFlip.\n'
        f'YOLOv8s: дообучение с весов COCO, изображения {yolo_args["imgsz"]} px, '
        f'{yolo_args["epochs"]} эпох, батч {yolo_args["batch"]}, стандартные аугментации Ultralytics '
        f'(mosaic, HSV, сдвиг, масштаб, отражение).'
    )
    pdf.chapter_body(description_text)

    # 2. Качество
    pdf.chapter_title('2. Выводы по метрикам качества (тестовая выборка)')
    better_map = 'YOLOv8s' if yolo['mAP'] > fcos['mAP'] else 'FCOS'
    quality_text = (
        f'МЕТРИКИ:\n'
        f'  FCOS     - mAP@0.5:0.95 = {fcos["mAP"]:.3f}, mAP@0.5 = {fcos["mAP_50"]:.3f}\n'
        f'  YOLOv8s  - mAP@0.5:0.95 = {yolo["mAP"]:.3f}, mAP@0.5 = {yolo["mAP_50"]:.3f}\n'
        f'Выше по mAP@0.5:0.95: {better_map} '
        f'(разница {abs(yolo["mAP"] - fcos["mAP"]):.3f}).\n\n'
        f'{quality_conclusion}'
    )
    pdf.chapter_body(quality_text)

    # 3. Скорость
    pdf.chapter_title('3. Выводы по скорости инференса')
    speed_text = (
        f'FPS (сквозной замер на одной видеокарте, 50 прогонов одного изображения):\n'
        f'  FCOS     - {fcos["FPS"]:.1f} FPS (изображения {fcos_cfg.img_scale[0]} px)\n'
        f'  YOLOv8s  - {yolo["FPS"]:.1f} FPS (изображения {yolo_args["imgsz"]} px)\n'
        f'YOLOv8s быстрее FCOS в {yolo["FPS"] / fcos["FPS"]:.1f} раза.\n\n'
        f'{speed_conclusion}'
    )
    pdf.chapter_body(speed_text)

    # 4. Графики
    pdf.add_image_section('4. Сравнение метрик', os.path.join(METRICS_DIR, 'metrics_comparison.png'),
                          'mAP@0.5:0.95, mAP@0.5 на тестовой выборке и FPS для обеих моделей.',
                          image_width=180)
    pdf.add_image_section('Ход обучения FCOS', os.path.join(FCOS_WORK_DIR, 'fcos_training_curves.png'),
                          'Функции потерь, learning rate и mAP на валидации по эпохам (MMDetection).',
                          image_width=180)
    pdf.add_image_section('Ход обучения YOLOv8s', os.path.join(YOLO_WORK_DIR, 'yolo_training_curves.png'),
                          'Функции потерь на train и val и метрики на валидации по эпохам (Ultralytics).',
                          image_width=180)

    # 5. Примеры детекций
    test_coco = load_coco(ANN_FILES['test'])
    id_to_name = {cat['id']: cat['name'] for cat in test_coco['categories']}
    file_to_id = {img['file_name']: img['id'] for img in test_coco['images']}
    for i, image_name in enumerate(example_images):
        objects = Counter(id_to_name[ann['category_id']] for ann in test_coco['annotations']
                          if ann['image_id'] == file_to_id[image_name])
        stats_text = (
            f'Файл: {image_name}\n'
            f'Слева - FCOS, справа - YOLOv8s (порог уверенности 0.5).\n'
            f'Объекты по разметке: ' + ', '.join(f'{name} x{count}' for name, count in objects.items())
        )
        pdf.add_image_section(f'5.{i + 1}. Пример детекции', make_comparison_image(image_name),
                              stats_text, image_width=180)

    pdf.output(output_path)
    print(f'Отчёт сохранён в {output_path}')


if __name__ == '__main__':
    build_report(
        quality_conclusion='Выводы о качестве моделей заполняются в ноутбуке после обучения.',
        speed_conclusion='Выводы о скорости моделей заполняются в ноутбуке после обучения.',
        example_images=['grass_desert_-_cow_3_1434_jpg.rf.110c73a314ed050344613c069fbc1328.jpg']
    )
