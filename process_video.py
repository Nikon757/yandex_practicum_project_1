"""
Покадровый инференс на видео - по уроку об обучении YOLO (play_game.py),
без игровой логики: каждый кадр проходит через детектор, рамки рисуются
общей функцией draw_boxes, кадр записывается в выходное видео.

Запуск из консоли:
    python process_video.py --model fcos --weights artifacts/fcos/best_coco_bbox_mAP_epoch_12.pth
    python process_video.py --model yolo --weights artifacts/yolo/yolov8s_minecraft/weights/best.pt
"""
import os
import argparse

import cv2
from tqdm import tqdm

from constants import FCOS_CONFIG, VIDEO_PATH, VIDEOS_DIR
from detection_models import device, predict, draw_boxes


def process_video(model_name, model, input_video_path, output_video_path, confidence_threshold=0.5):
    """
    Обрабатывает видео: детекция на каждом кадре и запись результата в новый файл.

    Args:
        model_name (str): Название модели ('FCOS' или 'YOLOv8s') - по нему выбирается API.
        model: Загруженная модель.
        input_video_path (str): Путь к исходному видео.
        output_video_path (str): Путь для сохранения видео с детекциями.
        confidence_threshold (float): Порог уверенности для отрисовки рамок.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видеофайл {input_video_path}")
        return

    # Получаем метаданные видео для корректной записи выходного файла
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Инициализируем объект для записи видео с кодеком 'mp4v'
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    with tqdm(total=total_frames, desc=f"Инференс {model_name}") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break  # Видео закончилось

            boxes, labels, scores = predict(model_name, model, frame)
            frame = draw_boxes(frame, boxes, labels, scores, threshold=confidence_threshold)

            # Записываем обработанный кадр в выходной видеофайл
            out.write(frame)
            pbar.update(1)

    # Освобождаем ресурсы, чтобы файлы корректно сохранились
    cap.release()
    out.release()
    print(f"\nОбработка завершена. Результат сохранен в файл: {output_video_path}")


def show_video_frames(video_paths, frame_index):
    """
    Показывает один и тот же кадр из нескольких видео рядом -
    чтобы сравнить детекции моделей в ноутбуке.

    Args:
        video_paths (dict): {название модели: путь к видео с детекциями}.
        frame_index (int): Номер кадра.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(video_paths), figsize=(12 * len(video_paths), 6))
    for ax, (model_name, video_path) in zip(axs, video_paths.items()):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        cap.release()

        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{model_name}, кадр {frame_index}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс детектора на видео")
    parser.add_argument("--model", type=str, required=True, choices=["fcos", "yolo"], help="Какую модель запускать")
    parser.add_argument("--weights", type=str, required=True, help="Путь к весам обученной модели")
    parser.add_argument("--input-video", type=str, default=VIDEO_PATH, help="Путь к исходному видеофайлу")
    parser.add_argument("--output-video", type=str, default=None, help="Путь для сохранения видео")
    parser.add_argument("--confidence", type=float, default=0.5, help="Порог уверенности")
    args = parser.parse_args()

    if args.model == "fcos":
        from mmdet.apis import init_detector
        model_name, model = "FCOS", init_detector(FCOS_CONFIG, args.weights, device=device)
    else:
        from ultralytics import YOLO
        model_name, model = "YOLOv8s", YOLO(args.weights)

    output_video = args.output_video or os.path.join(VIDEOS_DIR, f"{args.model}_inference.mp4")
    process_video(model_name, model, args.input_video, output_video, args.confidence)
