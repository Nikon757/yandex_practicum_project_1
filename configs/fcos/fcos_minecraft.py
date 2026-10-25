# Конфигурация дообучения FCOS на датасете персонажей Minecraft.
#
# За основу взята стандартная сборка FCOS (ResNet-50 caffe + FPN + голова с GroupNorm,
# расписание 1x). От базового конфига наследуем всё и переопределяем только то,
# что нужно для нашего датасета и для быстрого обучения на одной видеокарте.
#
# Параметры из задания даны в терминах MMDetection 2.x, здесь используется MMDetection 3.x:
#   img_scale                      -> Resize(scale=...)
#   samples_per_gpu / workers_per_gpu -> train_dataloader.batch_size / num_workers
#   runner(max_epochs)             -> train_cfg.max_epochs
#   checkpoint_interval            -> default_hooks.checkpoint.interval
#   fp16 = dict(loss_scale='dynamic') -> optim_wrapper(type='AmpOptimWrapper', loss_scale='dynamic')

_base_ = 'fcos_r50-caffe_fpn_gn-head_1x_coco.py'

# ----------------------------------------------------------------------------
# Классы датасета
# ----------------------------------------------------------------------------
# metainfo подставляется в CocoDataset вместо 80 классов COCO. Порядок совпадает
# с constants.CLASSES и с id категорий 1..17 в annotations.json (служебная
# категория 'minecraft-mobs' с id 0 в разметке не используется и сюда не входит).
metainfo = dict(
    classes=('bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox', 'frog', 'ghast',
             'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider', 'turtle', 'wolf', 'zombie'),
    palette=[(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
             (165, 42, 42), (255, 77, 255), (0, 226, 252)]
)
num_classes = len(metainfo['classes'])

# ----------------------------------------------------------------------------
# Модель
# ----------------------------------------------------------------------------
model = dict(
    # Веса backbone придут вместе со всей моделью из load_from,
    # поэтому отдельную загрузку ImageNet-весов ResNet отключаем
    backbone=dict(init_cfg=None),
    # Голова предсказывает 17 наших классов вместо 80 классов COCO
    bbox_head=dict(num_classes=num_classes)
)

# Стартуем с модели, обученной на COCO (чекпоинт из Model Zoo MMDetection).
# Слой классификации в голове не совпадает по размеру (80 -> 17 классов) -
# он инициализируется заново и обучается с нуля, остальные веса переносятся.
load_from = 'checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth'

# ----------------------------------------------------------------------------
# Данные
# ----------------------------------------------------------------------------
dataset_type = 'CocoDataset'
data_root = 'datasets/minecraft/'

# Изображения сжимаем до 640 по длинной стороне (с сохранением пропорций):
# так обучение идёт быстрее, но мелкие объекты становятся ещё мельче
img_scale = (640, 640)

# Аугментации в пайплайне простые: масштабирование и горизонтальное отражение
# (по заданию, без тяжёлых вроде Mosaic). Основную работу по «размножению» редких
# классов делает офлайн-скрипт augment_dataset.py (пайплайн Albumentations из урока
# о подготовке данных): его копии подмешиваются к обучающей выборке ниже.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# Обучающая выборка = исходный train + аугментированные копии изображений с редкими
# классами (какие классы и сколько копий - constants.RARE_CLASS_REPEATS). Копии лежат
# в images/train_augmented/, их COCO-разметку пишет augment_dataset.py.
# ConcatDataset объединяет два CocoDataset с одинаковым metainfo в один.
train_dataloader = dict(
    batch_size=4,   # samples_per_gpu=4 - эксперимент
    num_workers=4,  # workers_per_gpu=4
    dataset=dict(
        _delete_=True,  
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                metainfo=metainfo,
                ann_file='annotations/train/annotations.json',
                data_prefix=dict(img='images/train/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                metainfo=metainfo,
                ann_file='annotations/train_augmented/annotations.json',
                data_prefix=dict(img='images/train_augmented/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline),
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/valid/annotations.json',
        data_prefix=dict(img='images/valid/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test/annotations.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline))

# Метрики: путь до ground truth-разметки нужной выборки, classwise=True -
# помимо общего mAP печатать AP по каждому классу
val_evaluator = dict(
    ann_file=data_root + 'annotations/valid/annotations.json',
    classwise=True)
test_evaluator = dict(
    ann_file=data_root + 'annotations/test/annotations.json',
    classwise=True)

# ----------------------------------------------------------------------------
# Обучение
# ----------------------------------------------------------------------------
# Не более 12 эпох, валидация после каждой эпохи
train_cfg = dict(max_epochs=12, val_interval=1) 

# Оптимизатор и расписание lr - стандартные из базового конфига FCOS
# (SGD, lr=0.01, momentum=0.9, weight_decay=0.0001, прогрев 500 итераций,
# снижение lr в 10 раз на 8-й и 11-й эпохах). Добавляем только смешанную
# точность (fp16) с динамическим масштабированием loss - экономит память и время
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')

# Чекпоинт - раз в эпоху, храним два последних и лучший по mAP на валидации;
# лог в консоль - каждые 50 итераций
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2,
                    save_best='coco/bbox_mAP'),
    logger=dict(type='LoggerHook', interval=50))

# Папка для логов, чекпоинтов и метрик
work_dir = 'artifacts/fcos'
