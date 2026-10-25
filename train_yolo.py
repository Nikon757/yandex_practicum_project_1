"""
Дообучение YOLOv8 (Ultralytics) и графики метрик обучения из results.csv.

Функция train_yolo повторяет train_model из урока об обучении YOLO:
проверка data.yaml, загрузка предобученной модели, вызов model.train().
Скрипт можно запустить и из консоли:

    python train_yolo.py --data datasets/minecraft/data.yaml --epochs 30 --imgsz 640 --batch 16
"""
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

from constants import DATA_YAML, YOLO_PRETRAINED, YOLO_WORK_DIR, YOLO_EXPERIMENT

# Аугментации Ultralytics «на лету» (как в уроке об обучении YOLO): цветовые сдвиги HSV,
# случайный сдвиг и масштаб, горизонтальное отражение, mosaic. Редкие классы
# уже «размножены» офлайн скриптом augment_dataset.py, поэтому здесь ничего тяжёлого.
AUGMENTATIONS = {
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'translate': 0.1,
    'scale': 0.5,
    'fliplr': 0.5,
    'mosaic': 1.0,
}


def train_yolo(data_path=DATA_YAML, epochs=30, imgsz=640, batch=16, model_name=YOLO_PRETRAINED,
               project=YOLO_WORK_DIR, experiment_name=YOLO_EXPERIMENT, workers=2, patience=10):
    """
    Запускает дообучение YOLOv8 (как train_model из урока).

    Args:
        data_path (str): Путь к data.yaml.
        epochs (int): Количество эпох.
        imgsz (int): Размер изображения.
        batch (int): Размер батча.
        model_name (str): Предобученные веса (например, checkpoints/yolov8s.pt).
        project (str): Папка проекта Ultralytics (куда складываются эксперименты).
        experiment_name (str): Имя эксперимента - подпапка в project.
        workers (int): Количество потоков загрузки данных.
        patience (int): Эпох без улучшения до ранней остановки.

    Returns:
        Результаты обучения (объект метрик Ultralytics).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Файл {data_path} не найден!')

    model = YOLO(model_name)

    results = model.train(
        # Относительный путь Ultralytics ищет в своей глобальной папке datasets_dir,
        # поэтому передаём абсолютный - тогда пути внутри data.yaml
        # отсчитываются от папки самого data.yaml
        data=os.path.abspath(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=experiment_name,
        exist_ok=True,      # перезаписывать папку эксперимента при повторном запуске
        workers=workers,
        patience=patience,
        seed=42,            # фиксируем seed для воспроизводимости
        **AUGMENTATIONS
    )
    return results


# ----------------------------------------------------------------------------
# Логи обучения
# ----------------------------------------------------------------------------

def find_best_weights(project=YOLO_WORK_DIR, experiment_name=YOLO_EXPERIMENT):
    """Путь к лучшим весам обученной модели (папка эксперимента Ultralytics: project/name)."""
    weights = os.path.join(project, experiment_name, 'weights', 'best.pt')
    if not os.path.exists(weights):
        raise FileNotFoundError(f'Файл {weights} не найден - обучение ещё не проводилось')
    return weights


def load_yolo_results(project=YOLO_WORK_DIR, experiment_name=YOLO_EXPERIMENT):
    """Читает results.csv, который Ultralytics пишет после каждой эпохи."""
    results_file = os.path.join(project, experiment_name, 'results.csv')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f'Файл {results_file} не найден - обучение ещё не проводилось')

    results = pd.read_csv(results_file)
    # В заголовках results.csv есть лишние пробелы - убираем их
    results.columns = results.columns.str.strip()

    # При повторном запуске с exist_ok=True Ultralytics дописывает results.csv,
    # а не перезаписывает - берём строки только последнего запуска (с последней 1-й эпохи)
    last_run_start = results.index[results['epoch'] == 1][-1]
    return results.loc[last_run_start:].reset_index(drop=True)


def plot_yolo_metrics(results, save_path=None):
    """Строит графики loss (train / val) и метрик на валидации по эпохам."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('YOLOv8s: ход обучения', fontsize=16)

    for column in ['train/box_loss', 'train/cls_loss', 'train/dfl_loss']:
        axs[0].plot(results['epoch'], results[column], label=column)
    axs[0].set_title('Функции потерь (train)')

    for column in ['val/box_loss', 'val/cls_loss', 'val/dfl_loss']:
        axs[1].plot(results['epoch'], results[column], label=column)
    axs[1].set_title('Функции потерь (val)')

    metrics = {
        'metrics/mAP50-95(B)': 'mAP@0.5:0.95',
        'metrics/mAP50(B)': 'mAP@0.5',
        'metrics/precision(B)': 'precision',
        'metrics/recall(B)': 'recall',
    }
    for column, label in metrics.items():
        axs[2].plot(results['epoch'], results[column], marker='o', label=label)
    axs[2].set_title('Метрики на валидации')
    axs[2].set_ylim(0, 1)

    for ax in axs:
        ax.set_xlabel('Эпоха')
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Дообучение YOLOv8 на датасете Minecraft')
    parser.add_argument('--data', type=str, default=DATA_YAML, help='Путь к data.yaml')
    parser.add_argument('--epochs', type=int, default=30, help='Количество эпох обучения')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения')
    parser.add_argument('--batch', type=int, default=16, help='Размер батча')
    parser.add_argument('--model', type=str, default=YOLO_PRETRAINED, help='Предобученные веса')
    parser.add_argument('--name', type=str, default=YOLO_EXPERIMENT, help='Название эксперимента')
    args = parser.parse_args()

    train_yolo(
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_name=args.model,
        experiment_name=args.name
    )
