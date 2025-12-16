"""
Конфигурационный файл для приложения LiverSeg 3D.

Содержит пути к весам моделей и другие настройки конфигурации.
"""

import os
from typing import Dict, Optional

# Базовый путь к проекту
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Базовая директория для моделей - теперь та же папка, что и release (для поиска checkpoints внутри)
# Если нужно использовать переменную окружения, она будет иметь приоритет
MODEL_BASE_DIR = os.getenv('LIVERSEG_MODEL_DIR', BASE_DIR)

# Пути к директориям с моделями
# Теперь ищем сначала в папке checkpoints внутри release, затем в Final Models на 2 уровня выше (оригинальное расположение)
MODEL_DIRS = {
    'checkpoints': os.path.join(BASE_DIR, 'checkpoints'),  # Папка checkpoints внутри release
    'final_models': os.path.join(BASE_DIR, '..', '..', 'Final Models'),  # Папка Final Models на 2 уровня выше (оригинальное расположение)
    'venv': os.path.join(BASE_DIR, '.venv'),
}

# Пути к весам моделей YOLO
YOLO_MODEL_PATHS = {
    'best': os.path.join(MODEL_DIRS['final_models'], 'runs', 'segment', 'yolo_ts_liver', 'weights', 'best.pt'),
    'last': os.path.join(MODEL_DIRS['final_models'], 'runs', 'segment', 'yolo_ts_liver', 'weights', 'last.pt'),
}

# Пути к весам моделей U-Net
UNET_MODEL_PATHS = {
    'best': os.path.join(MODEL_DIRS['final_models'], 'UNet_best.pth'),
    'checkpoint': os.path.join(MODEL_DIRS['final_models'], 'UNet_best.pth'),
    'unet_dir': os.path.join(MODEL_DIRS['final_models'], 'UNet_best.pth'),
}

# Пути к весам моделей nnU-Net
NNUNET_MODEL_PATHS = {
    'best': os.path.join(MODEL_DIRS['final_models'], 'NnUNet_best.pth'),
    'checkpoint': os.path.join(MODEL_DIRS['final_models'], 'NnUNet_best.pth'),
}

# Пути к весам других моделей
OTHER_MODEL_PATHS = {
    'attention_unet': os.path.join(MODEL_DIRS['final_models'], 'UNet_best.pth'),  # Используем UNet как замену
    'resunet': os.path.join(MODEL_DIRS['final_models'], 'NnUNet_best.pth'),  # Используем NnUNet как замену
}

# Основные пути к моделям для использования в приложении
DEFAULT_MODEL_PATHS = {
    'YOLOv11': YOLO_MODEL_PATHS['best'],
    'U-Net': UNET_MODEL_PATHS['best'],
    'nnU-Net': NNUNET_MODEL_PATHS['best'],
}

# Настройки устройств
DEVICE_CONFIG = {
    'use_cuda': True,
    'cuda_device': 0,  # ID GPU для использования
    'fallback_to_cpu': True,  # Использовать CPU если CUDA недоступна
}

# Настройки инференса
INFERENCE_CONFIG = {
    'batch_size': 1,  # Размер батча для инференса
    'threshold': 0.5,  # Порог для бинаризации маски
    'use_tta': False,  # Test Time Augmentation
    'tta_transforms': ['horizontal_flip', 'vertical_flip'],  # Трансформации для TTA
}

# Настройки предобработки
PREPROCESSING_CONFIG = {
    'normalize': True,
    'clip_values': True,
    'clip_range': (-1000, 1000),  # Диапазон для клиппинга HU значений
    'resize': False,
    'target_size': (512, 512),  # Целевой размер для изменения размера
}

# Настройки постобработки
POSTPROCESSING_CONFIG = {
    'remove_small_objects': True,
    'min_object_size': 100,  # Минимальный размер объекта в пикселях
    'fill_holes': True,
    'morphology_ops': True,
    'kernel_size': 3,  # Размер ядра для морфологических операций
}

# Настройки визуализации
VISUALIZATION_CONFIG = {
    'colormap': 'jet',
    'opacity': 0.5,
    'show_axes': True,
    'background_color': 'white',
    'camera_position': 'iso',
}

# Настройки превью
PREVIEW_CONFIG = {
    'enabled': True,
    'default_resolution': 'original',  # 'original', 128, 256, 512
    'available_resolutions': ['original', 128, 256, 512],
    'cache_size': 10,  # Количество хранимых превью в кэше
    'use_preview_for_display': True,  # Использовать превью для отображения
    'downsample_method': 'bilinear',  # Метод даунсэмплинга ('bilinear', 'nearest', 'lanczos')
    'quality_level': 'high',  # Уровень качества ('low', 'medium', 'high')
}

# Настройки логирования
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': os.path.join(BASE_DIR, 'liverseg3d.log'),
    'max_file_size': 10 * 1024 * 1024,  # 10 MB
    'backup_count': 5,
}

# Настройки YOLO модели
YOLO_CONFIG = {
    'confidence_threshold': 0.1,  # Порог уверенности для детекций
    'iou_threshold': 0.7,  # Порог IoU для NMS
    'window_center': 40,  # Центр окна для HU нормализации
    'window_width': 400,  # Ширина окна для HU нормализации
}


def get_model_path(model_name: str, variant: str = 'default') -> Optional[str]:
    """
    Получение пути к весам модели.
    
    Args:
        model_name: Имя модели ('YOLOv11', 'U-Net', 'nnU-Net')
        variant: Вариант пути ('default', 'best', 'checkpoint', etc.)
        
    Returns:
        Путь к файлу весов или None если файл не найден
    """
    if model_name == 'YOLOv11':
        if variant == 'default' or variant == 'best':
            path = YOLO_MODEL_PATHS['best']
        elif variant == 'last':
            path = YOLO_MODEL_PATHS['last']
        else:
            path = None
    elif model_name == 'U-Net':
        if variant == 'default' or variant == 'best':
            path = UNET_MODEL_PATHS['best']
        elif variant == 'checkpoint':
            path = UNET_MODEL_PATHS['checkpoint']
        elif variant == 'unet_dir':
            path = UNET_MODEL_PATHS['unet_dir']
        else:
            path = None
    elif model_name == 'nnU-Net':
        if variant == 'default' or variant == 'best':
            path = NNUNET_MODEL_PATHS['best']
        elif variant == 'checkpoint':
            path = NNUNET_MODEL_PATHS['checkpoint']
        else:
            path = None
    else:
        path = None
    
    # Проверка существования файла
    if path and os.path.exists(path):
        return path
    
    # Если файл не найден, пробуем альтернативные пути
    return find_model_file(model_name)


def find_model_file(model_name: str) -> Optional[str]:
    """
    Поиск файла весов модели в различных директориях.
    
    Args:
        model_name: Имя модели
        
    Returns:
        Путь к файлу весов или None если файл не найден
    """
    # Возможные имена файлов
    possible_names = []
    
    if model_name == 'YOLOv11':
        possible_names = ['best.pt', 'last.pt', 'yolo_liver.pt', 'yolov11_liver.pt']
    elif model_name == 'U-Net':
        possible_names = ['UNet_best.pth', 'unet_liver.pth', 'best_model.pth', 'unet.pth']
    elif model_name == 'nnU-Net':
        possible_names = ['NnUNet_best.pth', 'nnunet_liver.pth', 'nnunet.pth']
    
    # Поиск в директориях
    for dir_name, dir_path in MODEL_DIRS.items():
        if not os.path.exists(dir_path):
            continue
            
        # Рекурсивный поиск
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file in possible_names:
                    return os.path.join(root, file)
    
    return None


def validate_model_paths() -> Dict[str, bool]:
    """
    Проверка существования файлов весов моделей.
    
    Returns:
        Словарь с результатами проверки для каждой модели
    """
    results = {}
    
    for model_name, path in DEFAULT_MODEL_PATHS.items():
        results[model_name] = os.path.exists(path)
        
        if not results[model_name]:
            # Пробуем найти альтернативный путь
            alt_path = find_model_file(model_name)
            results[model_name] = alt_path is not None
            if alt_path:
                DEFAULT_MODEL_PATHS[model_name] = alt_path
    
    return results


def get_device() -> str:
    """
    Определение устройства для вычислений.
    
    Returns:
        Строка с названием устройства ('cuda', 'cpu')
    """
    import torch
    
    if DEVICE_CONFIG['use_cuda'] and torch.cuda.is_available():
        return f"cuda:{DEVICE_CONFIG['cuda_device']}"
    elif DEVICE_CONFIG['fallback_to_cpu']:
        return 'cpu'
    else:
        raise RuntimeError("CUDA недоступна и fallback_to_cpu отключен")


def get_config() -> Dict:
    """
    Получение полной конфигурации.
    
    Returns:
        Словарь со всеми настройками
    """
    return {
        'model_paths': DEFAULT_MODEL_PATHS,
        'device': get_device(),
        'inference': INFERENCE_CONFIG,
        'preprocessing': PREPROCESSING_CONFIG,
        'postprocessing': POSTPROCESSING_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'preview': PREVIEW_CONFIG,
        'logging': LOGGING_CONFIG,
        'yolo': YOLO_CONFIG,
    }


# Инициализация и проверка путей при импорте
if __name__ == "__main__":
    # Проверка путей к моделям
    validation_results = validate_model_paths()
    
    print("Проверка путей к моделям:")
    for model_name, exists in validation_results.items():
        status = "[OK]" if exists else "[X]"
        print(f"  {status} {model_name}: {DEFAULT_MODEL_PATHS[model_name]}")
    
    # Вывод информации об устройстве
    print(f"\nУстройство для вычислений: {get_device()}")
    
    # Вывод конфигурации
    print("\nКонфигурация:")
    config = get_config()
    for section, settings in config.items():
        if section != 'model_paths':  # Уже вывели выше
            print(f"  {section}: {settings}")