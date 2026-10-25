"""
Дообучение FCOS средствами MMEngine / MMDetection и работа с логами обучения.

Запуск обучения из ноутбука делает то же, что tools/train.py: читает конфиг,
собирает Runner и вызывает runner.train(). Скрипт можно запустить и из консоли:

    python train_fcos.py --config configs/fcos/fcos_minecraft.py --work-dir artifacts/fcos
"""
import os
import glob
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.runner import Runner

from constants import FCOS_CONFIG, FCOS_WORK_DIR


def build_config(config_file=FCOS_CONFIG, work_dir=FCOS_WORK_DIR, resume=False):
    """
    Читает конфиг и применяет настройки конкретного запуска.

    Args:
        config_file (str): Путь к конфигу обучения.
        work_dir (str): Папка для логов, чекпоинтов и метрик.
        resume (bool): Продолжить обучение с последнего чекпоинта в work_dir.

    Returns:
        Config: Конфиг MMEngine.
    """
    cfg = Config.fromfile(config_file)
    cfg.work_dir = work_dir

    if resume:
        # При возобновлении веса и состояние оптимизатора берутся
        # из последнего чекпоинта в work_dir, а не из load_from
        cfg.resume = True
        cfg.load_from = None

    return cfg


def train_fcos(config_file=FCOS_CONFIG, work_dir=FCOS_WORK_DIR, resume=False):
    """
    Запускает дообучение FCOS.

    Returns:
        Runner: Объект Runner после обучения (в runner.model - обученная модель).
    """
    cfg = build_config(config_file, work_dir, resume)
    runner = Runner.from_cfg(cfg)
    runner.train()
    return runner


# ----------------------------------------------------------------------------
# Логи обучения
# ----------------------------------------------------------------------------

def find_scalars_log(work_dir=FCOS_WORK_DIR):
    """
    Находит файл с метриками последнего запуска.

    MMDetection 3.x пишет метрики в <work_dir>/<время запуска>/vis_data/scalars.json
    (аналог log.json из MMDetection 2.x): по одной JSON-строке на запись.
    """
    log_files = sorted(glob.glob(os.path.join(work_dir, '*', 'vis_data', 'scalars.json')))
    if not log_files:
        raise FileNotFoundError(f'В папке {work_dir} нет логов обучения (vis_data/scalars.json)')
    return log_files[-1]


def load_fcos_log(log_file):
    """
    Читает scalars.json и разделяет записи на обучающие и валидационные.

    Записи с обучения содержат значения loss и lr (step - номер итерации),
    записи с валидации - метрики COCO (step - номер эпохи).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_log, val_log).
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    train_log = pd.DataFrame([r for r in records if 'loss' in r])
    val_log = pd.DataFrame([r for r in records if 'coco/bbox_mAP' in r])
    return train_log, val_log


def plot_fcos_metrics(train_log, val_log, save_path=None):
    """Строит графики loss, learning rate и mAP по эпохам."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('FCOS: ход обучения', fontsize=16)

    for column in ['loss', 'loss_cls', 'loss_bbox', 'loss_centerness']:
        axs[0].plot(train_log['step'], train_log[column], label=column)
    axs[0].set_title('Функции потерь (train)')
    axs[0].set_xlabel('Итерация')
    axs[0].legend()

    axs[1].plot(train_log['step'], train_log['lr'])
    axs[1].set_title('Learning rate')
    axs[1].set_xlabel('Итерация')

    axs[2].plot(val_log['step'], val_log['coco/bbox_mAP'], marker='o', label='mAP@0.5:0.95')
    axs[2].plot(val_log['step'], val_log['coco/bbox_mAP_50'], marker='o', label='mAP@0.5')
    axs[2].plot(val_log['step'], val_log['coco/bbox_mAP_75'], marker='o', label='mAP@0.75')
    axs[2].set_title('Метрики на валидации')
    axs[2].set_xlabel('Эпоха')
    axs[2].set_ylim(0, 1)
    axs[2].legend()

    for ax in axs:
        ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()


def find_best_checkpoint(work_dir=FCOS_WORK_DIR):
    """Возвращает путь к лучшему по mAP чекпоинту (best_coco_bbox_mAP_epoch_N.pth)."""
    checkpoints = sorted(glob.glob(os.path.join(work_dir, 'best_coco_bbox_mAP_epoch_*.pth')),
                         key=os.path.getmtime)
    if not checkpoints:
        raise FileNotFoundError(f'В папке {work_dir} нет чекпоинта best_coco_bbox_mAP_epoch_*.pth')
    return checkpoints[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Дообучение FCOS на датасете Minecraft')
    parser.add_argument('--config', type=str, default=FCOS_CONFIG, help='Путь к конфигу')
    parser.add_argument('--work-dir', type=str, default=FCOS_WORK_DIR, help='Папка для результатов')
    parser.add_argument('--resume', action='store_true', help='Продолжить с последнего чекпоинта')
    args = parser.parse_args()

    train_fcos(config_file=args.config, work_dir=args.work_dir, resume=args.resume)
