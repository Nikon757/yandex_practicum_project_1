"""
Оценка обученных моделей на тестовой выборке и сводная таблица сравнения.

- YOLO: evaluate_yolo повторяет evaluate_model из урока об обучении YOLO
  (model.val на нужной выборке, метрики из metrics.box).
- FCOS: evaluate_fcos делает то же, что tools/test.py из MMDetection -
  собирает Runner по конфигу с обученным чекпоинтом и вызывает runner.test().
- compare_models собирает mAP, mAP@0.5 и FPS обеих моделей в таблицу
  и сохраняет её в artifacts/metrics/metrics_comparison.csv.

Запуск из консоли:
    python evaluate.py --fcos artifacts/fcos/best_coco_bbox_mAP_epoch_12.pth \\
                       --yolo artifacts/yolo/yolov8s_minecraft/weights/best.pt
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mmengine.config import Config
from mmengine.runner import Runner
from ultralytics import YOLO

from constants import CLASSES, FCOS_CONFIG, FCOS_WORK_DIR, DATA_YAML, YOLO_WORK_DIR, METRICS_DIR


def evaluate_fcos(checkpoint_file, config_file=FCOS_CONFIG, work_dir=FCOS_WORK_DIR):
    """
    Считает метрики COCO на тестовой выборке для FCOS.

    Args:
        checkpoint_file (str): Путь к обученному чекпоинту.
        config_file (str): Путь к конфигу модели (в нём заданы test_dataloader и test_evaluator).
        work_dir (str): Папка для логов оценки (подпапка test/).

    Returns:
        dict: {'mAP': mAP@0.5:0.95, 'mAP_50': mAP@0.5, 'per_class': {класс: AP}}.
    """
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Чекпоинт {checkpoint_file} не найден!")

    cfg = Config.fromfile(config_file)
    cfg.load_from = checkpoint_file
    cfg.work_dir = os.path.join(work_dir, 'test')

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    print("\nРезультаты оценки FCOS:")
    print(f"  mAP@0.5: {metrics['coco/bbox_mAP_50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['coco/bbox_mAP']:.4f}")

    # CocoMetric с classwise=True кладёт AP каждого класса в ключ coco/<класс>_precision;
    # у классов, которых нет в тестовой разметке, значение nan - их пропускаем
    per_class = {class_name: metrics[f'coco/{class_name}_precision'] for class_name in CLASSES
                 if not np.isnan(metrics[f'coco/{class_name}_precision'])}

    return {'mAP': metrics['coco/bbox_mAP'], 'mAP_50': metrics['coco/bbox_mAP_50'], 'per_class': per_class}


def evaluate_yolo(model_path, data_path=DATA_YAML, split='test'):
    """
    Оценивает модель YOLO на выбранной выборке (evaluate_model из урока).

    Args:
        model_path (str): Путь к обученной модели.
        data_path (str): Путь к data.yaml.
        split (str): Набор для оценки (test/val/train).

    Returns:
        dict: {'mAP': mAP@0.5:0.95, 'mAP_50': mAP@0.5, 'per_class': {класс: AP}}.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель {model_path} не найдена!")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл {data_path} не найден!")

    model = YOLO(model_path)

    metrics = model.val(
        data=os.path.abspath(data_path),  # пути внутри data.yaml - от папки самого файла
        split=split,
        project=YOLO_WORK_DIR,            # графики оценки - рядом с результатами обучения
        name=f'val_{split}',
        exist_ok=True
    )

    print("\nРезультаты оценки YOLO:")
    print(f"  mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")

    print(f"  Точность (средняя): {metrics.box.p.mean():.4f}")
    print(f"  Полнота (средняя): {metrics.box.r.mean():.4f}")

    # metrics.box.ap - AP@0.5:0.95 по классам, ap_class_index - их индексы
    per_class = {CLASSES[int(class_index)]: float(ap)
                 for class_index, ap in zip(metrics.box.ap_class_index, metrics.box.ap)}

    return {'mAP': metrics.box.map, 'mAP_50': metrics.box.map50, 'per_class': per_class}


def compare_models(metrics, fps_results, save_path):
    """
    Собирает таблицу сравнения моделей и сохраняет её в CSV.

    Args:
        metrics (dict): {название модели: результат evaluate_*}.
        fps_results (dict): {название модели: FPS} - результат measure_fps.
        save_path (str): Куда сохранить CSV.

    Returns:
        pd.DataFrame: Таблица с колонками mAP, mAP_50, FPS.
    """
    table = pd.DataFrame({
        model_name: {'mAP': model_metrics['mAP'], 'mAP_50': model_metrics['mAP_50'], 'FPS': fps_results[model_name]}
        for model_name, model_metrics in metrics.items()
    }).T.round(4)
    table.index.name = 'model'

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    table.to_csv(save_path)
    print(f'Таблица сравнения сохранена в {save_path}')
    return table


def plot_comparison(table, save_path):
    """Столбчатые диаграммы для визуального сравнения моделей по каждой метрике."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Сравнение FCOS и YOLOv8s на тестовой выборке', fontsize=16)

    for ax, column in zip(axs, ['mAP', 'mAP_50', 'FPS']):
        bars = ax.bar(table.index, table[column], color=['steelblue', 'darkorange'])
        ax.bar_label(bars, fmt='%.3f' if column != 'FPS' else '%.1f')
        ax.set_title(column)
        ax.set_ylim(0, table[column].max() * 1.25)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def training_summary(fcos_val_log, yolo_log):
    """
    Сводка по обучению обеих моделей: сколько эпох, на какой эпохе лучший результат
    и какие метрики на валидации он дал.

    Args:
        fcos_val_log (pd.DataFrame): Валидационные записи из load_fcos_log.
        yolo_log (pd.DataFrame): Таблица из load_yolo_results.
    """
    fcos_best = fcos_val_log.loc[fcos_val_log['coco/bbox_mAP'].idxmax()]
    yolo_best = yolo_log.loc[yolo_log['metrics/mAP50-95(B)'].idxmax()]

    return pd.DataFrame({
        'FCOS': {
            'эпох обучения': len(fcos_val_log),
            'лучшая эпоха': int(fcos_best['step']),
            'mAP@0.5:0.95': round(fcos_best['coco/bbox_mAP'], 3),
            'mAP@0.5': round(fcos_best['coco/bbox_mAP_50'], 3),
        },
        'YOLOv8s': {
            'эпох обучения': len(yolo_log),
            'лучшая эпоха': int(yolo_best['epoch']),
            'mAP@0.5:0.95': round(yolo_best['metrics/mAP50-95(B)'], 3),
            'mAP@0.5': round(yolo_best['metrics/mAP50(B)'], 3),
        },
    }).T


def plot_training_comparison(fcos_val_log, yolo_log, save_path):
    """Метрики на валидации по эпохам для обеих моделей на одних осях."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Качество на валидации по эпохам', fontsize=16)

    pairs = [('mAP@0.5:0.95', 'coco/bbox_mAP', 'metrics/mAP50-95(B)'),
             ('mAP@0.5', 'coco/bbox_mAP_50', 'metrics/mAP50(B)')]
    for ax, (title, fcos_column, yolo_column) in zip(axs, pairs):
        ax.plot(fcos_val_log['step'], fcos_val_log[fcos_column], marker='o', label='FCOS')
        ax.plot(yolo_log['epoch'], yolo_log[yolo_column], marker='o', label='YOLOv8s')
        ax.set_title(title)
        ax.set_xlabel('Эпоха')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def per_class_table(metrics):
    """
    Таблица AP@0.5:0.95 по классам для обеих моделей (только классы, которые есть в тесте).

    Args:
        metrics (dict): {название модели: результат evaluate_*}.
    """
    table = pd.DataFrame({name: model_metrics['per_class'] for name, model_metrics in metrics.items()})
    table['разница'] = table['YOLOv8s'] - table['FCOS']
    table.index.name = 'class'
    return table.round(3).sort_values('YOLOv8s', ascending=False)


def plot_per_class(table, save_path):
    """Столбчатая диаграмма AP по классам: FCOS против YOLOv8s."""
    df = table.drop(columns='разница').reset_index().melt(
        id_vars='class', var_name='model', value_name='AP')

    plt.figure(figsize=(12, 5))
    sns.set_theme(style='whitegrid')
    ax = sns.barplot(data=df, x='class', y='AP', hue='model', palette='viridis')
    ax.set_title('AP@0.5:0.95 по классам на тестовой выборке', fontsize=16)
    ax.set_xlabel('Класс', fontsize=12)
    ax.set_ylabel('AP@0.5:0.95', fontsize=12)
    ax.legend(title='')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Оценка FCOS и YOLOv8 на тестовой выборке')
    parser.add_argument('--fcos', type=str, required=True, help='Путь к чекпоинту FCOS')
    parser.add_argument('--yolo', type=str, required=True, help='Путь к весам YOLO (best.pt)')
    args = parser.parse_args()

    print('FCOS:', evaluate_fcos(args.fcos))
    print('YOLOv8s:', evaluate_yolo(args.yolo))
