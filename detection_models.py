"""
Единый интерфейс инференса для FCOS (MMDetection) и YOLOv8 (Ultralytics)
и отрисовка результатов - по уроку о сравнении моделей.

У двух фреймворков разные API, поэтому их вывод приводится к одному виду.
Каждая функция predict_* принимает изображение numpy BGR (как читает cv2)
и возвращает кортеж (boxes, labels, scores):
    boxes  - np.ndarray [N, 4], координаты [x_min, y_min, x_max, y_max] в пикселях;
    labels - np.ndarray [N], индексы классов (0..16 в порядке CLASSES);
    scores - np.ndarray [N], уверенность каждой детекции.
"""
import os
import time

import cv2
import torch
import matplotlib.pyplot as plt
from mmdet.apis import inference_detector

from constants import CLASSES, COLORS
from eda import get_image_files

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_fcos(model, image):
    """
    Инференс FCOS на одном изображении.

    Порог уверенности задаётся в конфиге модели (test_cfg.score_thr = 0.05),
    так что на выходе есть и слабые детекции - их отсекаем при отрисовке.
    """
    result = inference_detector(model, image)
    instances = result.pred_instances.cpu()
    return instances.bboxes.numpy(), instances.labels.numpy(), instances.scores.numpy()


def predict_yolo(model, image):
    """
    Инференс YOLO на одном изображении.

    conf намеренно низкий (как score_thr у FCOS), чтобы обе модели
    отдавали полный список кандидатов, а порог применялся одинаково при отрисовке.
    """
    result = model.predict(image, imgsz=640, conf=0.05, verbose=False)[0]
    boxes = result.boxes
    return (boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy())


def predict(model_name, model, image):
    """
    Инференс любой из двух моделей: нужный API выбирается по имени модели
    (так же различали модели в measure_fps из урока).
    """
    if 'YOLO' in model_name:
        return predict_yolo(model, image)
    return predict_fcos(model, image)


def draw_boxes(image, boxes, labels, scores, threshold=0.5, show_text=True):
    """
    Отрисовывает рамки с подписью «класс: уверенность» (функция draw_boxes из урока).

    Отличие от урока: на вход и на выход - изображение numpy BGR, а не PIL,
    чтобы результат можно было сразу сохранить через cv2.imwrite или записать в видео.
    """
    image_np = image.copy()
    for i, box in enumerate(boxes):
        if scores[i] > threshold:
            class_index = int(labels[i])
            label_text = CLASSES[class_index]

            color = [int(c) for c in COLORS[class_index]]
            x_min, y_min, x_max, y_max = map(int, box)

            # Адаптивный размер шрифта
            box_width = x_max - x_min
            font_scale = max(0.4, min(1.0, box_width / 200.0))
            thickness = max(1, int(font_scale * 2))

            # Полный текст подписи
            display_text = f"{label_text}: {scores[i]:.2f}"

            # Расчёт размера текстового блока
            (text_width, text_height), baseline = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Позиция текста
            text_x = x_min
            text_y = y_min - 5

            # Сдвигаем текст, если он выходит за пределы изображения
            if text_y < text_height:
                text_y = y_min + text_height + 5

            # Отрисовка фона для текста
            if show_text:
                cv2.rectangle(
                    image_np,
                    (text_x, text_y - text_height - baseline),
                    (text_x + text_width, text_y + baseline),
                    color,
                    cv2.FILLED
                )

                cv2.putText(
                    image_np,
                    display_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )

            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), color, 2)

    return image_np


def run_and_visualize(image_path, models, threshold=0.5, show_text=True):
    """
    Прогоняет изображение через все модели и показывает результаты рядом
    (как run_and_visualize из урока о сравнении моделей).

    Args:
        image_path (str): Путь к изображению.
        models (dict): {название модели: модель}.
        threshold (float): Порог уверенности для отрисовки.
        show_text (bool): Рисовать ли подписи классов.
    """
    image = cv2.imread(image_path)

    fig, axs = plt.subplots(1, len(models), figsize=(12 * len(models), 7))
    for ax, (model_name, model) in zip(axs, models.items()):
        boxes, labels, scores = predict(model_name, model, image)
        image_drawn = draw_boxes(image, boxes, labels, scores, threshold=threshold, show_text=show_text)
        ax.imshow(cv2.cvtColor(image_drawn, cv2.COLOR_BGR2RGB))
        ax.set_title(model_name)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def generate_visualizations(model_name, model, source_dir, output_dir, threshold=0.5):
    """
    Прогоняет модель по всем изображениям папки и сохраняет картинки с рамками
    (как generate_visualizations из урока об анализе трафика, но рамки рисует
    общая для обеих моделей функция draw_boxes).

    Args:
        model_name (str): Название модели.
        model: Загруженная модель.
        source_dir (str): Папка с исходными изображениями.
        output_dir (str): Папка для сохранения аннотированных изображений.
        threshold (float): Порог уверенности для отрисовки.
    """
    os.makedirs(output_dir, exist_ok=True)

    image_names = get_image_files(source_dir)
    for image_name in image_names:
        image = cv2.imread(os.path.join(source_dir, image_name))
        boxes, labels, scores = predict(model_name, model, image)
        cv2.imwrite(os.path.join(output_dir, image_name),
                    draw_boxes(image, boxes, labels, scores, threshold=threshold))

    print(f'{model_name}: {len(image_names)} изображений с детекциями сохранены в {output_dir}')


def measure_fps(models_dict, image, num_runs=50):
    """
    Универсальная функция для измерения и сравнения FPS нескольких моделей
    (measure_fps из урока о сравнении моделей). Замеряется полный цикл
    инференса: предобработка, прямой проход и постобработка.

    Args:
        models_dict (dict): Словарь, где ключ - название модели, значение - сама модель.
        image (np.ndarray): Тестовое изображение (BGR).
        num_runs (int): Количество запусков для усреднения.
    """
    results = {}

    # Прогрев: первый прогон на GPU всегда дороже (инициализация памяти, компиляция ядер)
    for model_name, model in models_dict.items():
        predict(model_name, model, image)

    for model_name, model in models_dict.items():
        start_time = time.time()
        for _ in range(num_runs):
            predict(model_name, model, image)
        end_time = time.time()

        fps = num_runs / (end_time - start_time)
        results[model_name] = fps

    # Вывод результатов
    print("\n" + "=" * 20 + " Результаты измерения End-to-End FPS " + "=" * 20)
    for name, fps in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{name:<15}: {fps:.2f} FPS")

    return results
