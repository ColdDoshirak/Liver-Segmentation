"""
Модуль для загрузки и обработки DICOM файлов.

Этот модуль предоставляет классы и функции для работы с DICOM данными,
включая загрузку серий, конвертацию в numpy массивы и предобработку
для различных моделей машинного обучения.
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import List, Dict, Tuple, Optional, Union
import logging
import time
from functools import lru_cache

# Импортируем функции из mask_utils для работы с масками
try:
    from .mask_utils import apply_mask_to_volume, create_masked_slice, create_contour_overlay, create_transparent_overlay, smooth_mask, extract_largest_component
except ImportError:
    from mask_utils import apply_mask_to_volume, create_masked_slice, create_contour_overlay, create_transparent_overlay, smooth_mask, extract_largest_component

# Импорт монитора производительности
try:
    from .performance_monitor import performance_monitor, performance_timer
except ImportError:
    # Для локального тестирования
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from performance_monitor import performance_monitor, performance_timer

# Импорт конфигурации для параметров windowing
try:
    from InferenceAnd3D.config import YOLO_CONFIG
except ImportError:
    try:
        from config import YOLO_CONFIG
    except ImportError:
        YOLO_CONFIG = {'window_center': 40, 'window_width': 400}

# Настройка логирования с поддержкой кириллицы и Unicode
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DicomLoader:
    """
    Класс для загрузки и обработки DICOM файлов.
    
    Предоставляет методы для работы с DICOM сериями, включая загрузку,
    получение информации и конвертацию в различные форматы.
    """
    
    def __init__(self):
        """Инициализация загрузчика DICOM."""
        self.reader = sitk.ImageSeriesReader()
        
    def get_dicom_series(self, folder_path: str) -> List[str]:
        """
        Получить список ID DICOM серий в указанной папке.
        
        Args:
            folder_path (str): Путь к папке с DICOM файлами
            
        Returns:
            List[str]: Список ID DICOM серий
            
        Raises:
            FileNotFoundError: Если папка не существует
            ValueError: Если в папке нет DICOM файлов
        """
        # Проверка существования папки
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Папка не существует: {folder_path}")
            
        # Проверка, что путь является директорией
        if not os.path.isdir(folder_path):
            raise ValueError(f"Указанный путь не является директорией: {folder_path}")
            
        # Получение списка ID серий
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
        
        if not series_ids:
            raise ValueError(f"В папке {folder_path} не найдено DICOM серий")
            
        logger.info(f"Найдено {len(series_ids)} DICOM серий в папке {folder_path}")
        return list(series_ids)
    
    def load_series(self, folder_path: str, series_id: str) -> sitk.Image:
        """
        Загрузить выбранную DICOM серию.
        
        Args:
            folder_path (str): Путь к папке с DICOM файлами
            series_id (str): ID серии для загрузки
            
        Returns:
            sitk.Image: Загруженное изображение SimpleITK
            
        Raises:
            FileNotFoundError: Если папка не существует
            ValueError: Если серия не найдена или не может быть загружена
        """
        # Проверка существования папки
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Папка не существует: {folder_path}")
            
        # Получение путей к файлам серии
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder_path, series_id)
        
        if not series_file_names:
            raise ValueError(f"Серия с ID {series_id} не найдена в папке {folder_path}")
            
        try:
            # Настройка и выполнение чтения
            self.reader.SetFileNames(series_file_names)
            image = self.reader.Execute()
            logger.info(f"Успешно загружена серия {series_id} из {len(series_file_names)} файлов")
            return image
        except Exception as e:
            raise ValueError(f"Ошибка при загрузке серии {series_id}: {str(e)}")
    
    def get_series_info(self, folder_path: str, series_id: str) -> Dict[str, Union[str, int, float, Tuple]]:
        """
        Получить информацию о DICOM серии.
        
        Args:
            folder_path (str): Путь к папке с DICOM файлами
            series_id (str): ID серии
            
        Returns:
            Dict[str, Union[str, int, float, Tuple]]: Словарь с информацией о серии
            
        Raises:
            FileNotFoundError: Если папка не существует
            ValueError: Если серия не найдена
        """
        try:
            # Загрузка серии
            image = self.load_series(folder_path, series_id)
            
            # Получение основной информации
            size = image.GetSize()
            spacing = image.GetSpacing()
            origin = image.GetOrigin()
            direction = image.GetDirection()
            
            # Получение метаданных
            metadata = {}
            for key in image.GetMetaDataKeys():
                metadata[key] = image.GetMetaData(key)
                
            # Формирование результата
            info = {
                'series_id': series_id,
                'dimensions': size,
                'num_slices': size[2] if len(size) > 2 else 1,
                'voxel_size_mm': spacing,
                'voxel_volume_mm3': spacing[0] * spacing[1] * spacing[2] if len(spacing) >= 3 else spacing[0] * spacing[1],
                'origin': origin,
                'direction': direction,
                'pixel_type': image.GetPixelIDTypeAsString(),
                'metadata': metadata
            }
            
            return info
        except Exception as e:
            raise ValueError(f"Ошибка при получении информации о серии {series_id}: {str(e)}")
    
    @performance_timer("dicom.convert_to_numpy")
    def convert_to_numpy(self, sitk_image: sitk.Image) -> np.ndarray:
        """
        Конвертировать SimpleITK.Image в numpy массив с оптимизацией памяти.
        
        Args:
            sitk_image (sitk.Image): Изображение SimpleITK
            
        Returns:
            np.ndarray: Numpy массив с данными изображения
            
        Raises:
            ValueError: Если изображение не может быть конвертировано
        """
        try:
            # Оптимизированная конвертация в numpy массив с представлением вместо копии
            image_array = sitk.GetArrayFromImage(sitk_image)
            
            # Проверка типа данных и оптимизация при необходимости
            if image_array.dtype == np.float64:
                # Конвертируем в float32 для экономии памяти без потери точности
                image_array = image_array.astype(np.float32)
            
            logger.info(f"Изображение конвертировано в numpy массив размером {image_array.shape}")
            return image_array
        except Exception as e:
            raise ValueError(f"Ошибка при конвертации изображения в numpy массив: {str(e)}")
    
    def get_voxel_volume(self, sitk_image: sitk.Image) -> float:
        """
        Рассчитать объем одного вокселя в мм³.
        
        Args:
            sitk_image (sitk.Image): Изображение SimpleITK
            
        Returns:
            float: Объем одного вокселя в мм³
            
        Raises:
            ValueError: Если не удалось получить размеры вокселя
        """
        try:
            spacing = sitk_image.GetSpacing()
            if len(spacing) >= 3:
                volume = spacing[0] * spacing[1] * spacing[2]
            else:
                volume = spacing[0] * spacing[1]
            logger.info(f"Объем вокселя: {volume} мм³")
            return volume
        except Exception as e:
            raise ValueError(f"Ошибка при расчете объема вокселя: {str(e)}")


@lru_cache(maxsize=32)
def _cached_normalize_params(image_hash: int, model_type: str) -> Tuple[float, float, float, float]:
    """
    Кэшированные параметры нормализации для изображения.
    
    Args:
        image_hash: Хэш изображения для кэширования
        model_type: Тип модели
        
    Returns:
        Кортеж с параметрами нормализации (min_val, max_val, mean_val, std_val)
    """
    # Эта функция будет использоваться внутри preprocess_for_model
    # для кэширования параметров нормализации
    return (0.0, 1.0, 0.0, 1.0)  # Значения по умолчанию

@performance_timer("dicom.preprocess_for_model")
def preprocess_for_model(image_array: np.ndarray, model_type: str) -> np.ndarray:
    """
    Оптимизированная предобработка изображения для различных моделей.

    Args:
        image_array (np.ndarray): Входное изображение в формате numpy
        model_type (str): Тип модели ('YOLO', 'U-Net', 'nnU-Net')

    Returns:
        np.ndarray: Предобработанное изображение

    Raises:
        ValueError: Если указан неподдерживаемый тип модели
    """
    # Проверка входных данных
    if image_array is None or image_array.size == 0:
        logger.error("Получено пустое изображение для предобработки")
        raise ValueError("Пустое изображение")
    
    # Оптимизированная конвертация типа данных
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)
    
    # Получение параметров windowing из конфигурации
    window_center = YOLO_CONFIG.get('window_center', 40)
    window_width = YOLO_CONFIG.get('window_width', 40)
    min_hu = window_center - window_width // 2
    max_hu = window_center + window_width // 2
    
    # Целевой размер для моделей U-Net и nnU-Net (обучены на 256x256)
    target_size = (256, 256)
    
    if model_type.lower() in ['yolo', 'yolov11']:
        # Для YOLO НЕ применяем нормализацию, только конвертацию в float32
        # Windowing и нормализацию HU выполнит сам YOLO в model_manager.py
        processed = image_array
        
        # Проверка размерности и корректировка для YOLO
        if len(processed.shape) == 3:
            # Для 3D изображений не добавляем канал здесь, так как YOLO работает с 2D
            # Метод _predict_2d в model_manager.py будет обрабатывать каждый срез отдельно
            pass
        elif len(processed.shape) == 2:
            # Оптимизированное конвертирование в 3 канала для YOLO
            # Используем np.broadcast_to вместо np.stack для экономии памяти
            processed = np.broadcast_to(processed[:, :, np.newaxis], processed.shape + (3,))
            logger.info(f"Изображение конвертировано из 1 канала в 3 канала для YOLO: {image_array.shape} -> {processed.shape}")
        else:
            logger.error(f"Неподдерживаемая размерность изображения для YOLO: {processed.shape}")
            raise ValueError(f"Неподдерживаемая размерность изображения: {processed.shape}")
            
    elif model_type.lower() in ['u-net', 'unet', 'nnunet', 'nn-u-net', 'nnu-net'] or 'nnu' in model_type.lower():
        # Общая предобработка для U-Net и nnU-Net (windowing + ресайз)
        processed = image_array
        
        # Windowing (clip HU значений)
        processed = np.clip(processed, min_hu, max_hu)
        
        # Линейное преобразование в диапазон [0, 255] как в ноутбуке
        processed = (processed - min_hu) / (max_hu - min_hu) * 255.0
        
        # Конвертация в uint8 (экономия памяти)
        processed = processed.astype(np.uint8)
        
        # Ресайз до 256x256, если размеры не совпадают
        # Обработка 2D и 3D изображений
        original_shape = processed.shape
        if len(original_shape) == 2:
            h, w = original_shape
            if (h, w) != target_size:
                try:
                    import cv2
                    processed = cv2.resize(processed, target_size, interpolation=cv2.INTER_LINEAR)
                except ImportError:
                    from scipy import ndimage
                    zoom_factors = (target_size[0] / h, target_size[1] / w)
                    processed = ndimage.zoom(processed, zoom_factors, order=1)
                logger.info(f"Ресайз изображения 2D: {original_shape} -> {processed.shape}")
        elif len(original_shape) == 3:
            # 3D объем: ресайз каждого среза отдельно
            d, h, w = original_shape
            if (h, w) != target_size:
                try:
                    import cv2
                    resized_slices = []
                    for i in range(d):
                        slice_img = processed[i]
                        resized_slice = cv2.resize(slice_img, target_size, interpolation=cv2.INTER_LINEAR)
                        resized_slices.append(resized_slice)
                    processed = np.stack(resized_slices, axis=0)
                except ImportError:
                    from scipy import ndimage
                    zoom_factors = (1, target_size[0] / h, target_size[1] / w)
                    processed = ndimage.zoom(processed, zoom_factors, order=1)
                logger.info(f"Ресайз изображения 3D: {original_shape} -> {processed.shape}")
        
        # Нормализация в диапазон [0, 1] для подачи в модель
        processed = processed.astype(np.float32) / 255.0
        
        # Раздельная нормализация для U-Net и nnU-Net
        if model_type.lower() in ['u-net', 'unet']:
            # Для U-Net дополнительная минимаксная нормализация (уже в [0,1])
            pass
        else:
            # Для nnU-Net Z-score нормализация (после преобразования в [0,1])
            mean_val, std_val = processed.mean(), processed.std()
            if std_val > 0:
                processed = (processed - mean_val) / std_val
            else:
                processed = np.zeros_like(processed)
        
        # Для U-Net и nnU-Net не добавляем канал здесь, так как модель ожидает 2D срезы
        # Канал будет добавлен в методе _predict_2d в model_manager.py
        
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}. Поддерживаемые типы: YOLO, U-Net, nnU-Net")
    
    logger.info(f"Изображение предобработано для модели {model_type}, финальная форма: {processed.shape}")
    return processed


@performance_timer("dicom.postprocess_mask")
def postprocess_mask(mask_array: np.ndarray, target_shape: Optional[tuple] = None,
                     prob_threshold: float = 0.5, min_object_size: Optional[int] = None,
                     model_type: Optional[str] = None) -> np.ndarray:
    """
    Улучшенная постобработка маски сегментации печени.

    Args:
        mask_array (np.ndarray): Входная маска сегментации
        target_shape (Optional[tuple]): Целевая форма для выравнивания размеров
        prob_threshold (float): Порог бинаризации для вероятностных масок (0-1). По умолчанию 0.5.
        min_object_size (Optional[int]): Минимальный размер объекта в вокселях.
            Если None, автоматически определяется на основе размера изображения.
        model_type (Optional[str]): Тип модели ('YOLO', 'U-Net', 'nnU-Net') для адаптивной обработки

    Returns:
        np.ndarray: Постобработанная маска
    """
    # Логирование входной маски
    initial_nonzero = np.sum(mask_array > 0) if mask_array is not None else 0
    initial_shape = mask_array.shape if mask_array is not None else None
    logger.info(f"Постобработка маски: форма={initial_shape}, "
                f"ненулевых вокселей={initial_nonzero}, порог={prob_threshold}, модель={model_type}")
    
    # Установка адаптивного порога для разных моделей
    if model_type is not None:
        if model_type.lower() in ['yolo', 'yolov11']:
            prob_threshold = 0.3  # Более низкий порог для YOLO
        elif model_type.lower() in ['u-net', 'unet']:
            prob_threshold = 0.5  # Стандартный порог для U-Net
        elif model_type.lower() in ['nnunet', 'nn-u-net', 'nnu-net']:
            prob_threshold = 0.4  # Средний порог для nnU-Net
    
    # Проверка на пустую маску
    if mask_array is None or mask_array.size == 0:
        logger.warning("Получена пустая маска на вход постобработки")
        if target_shape is not None:
            return np.zeros(target_shape, dtype=np.uint8)
        else:
            return np.zeros((512, 512), dtype=np.uint8) if mask_array is None else np.zeros_like(mask_array)
    
    # Сохраняем исходную форму для возможного выравнивания
    original_shape = mask_array.shape
    
    # Проверка, является ли маска уже бинарной (0/1) в исходном типе
    unique_values = np.unique(mask_array)
    # Учитываем возможные значения 0 и 1 в любом типе данных (float, int)
    is_already_binary = (
        len(unique_values) <= 2 and
        np.all(np.isin(unique_values, [0, 1]))
    )
    
    if is_already_binary:
        logger.info(f"Маска уже бинарная, пропускаем бинаризацию. Уникальные значения: {unique_values}")
        processed = mask_array.astype(np.uint8)
    else:
        # Определяем, является ли маска вероятностной (значения в диапазоне [0, 1])
        min_val, max_val = mask_array.min(), mask_array.max()
        if max_val <= 1.0 and min_val >= 0.0:
            # Пороговая бинаризация для вероятностной маски
            threshold = prob_threshold
            processed = (mask_array > threshold).astype(np.uint8)
            logger.info(f"Применен порог {threshold} для вероятностной маски")
        else:
            # Иначе используем порог на основе среднего значения (для масок с другими диапазонами)
            threshold = mask_array.mean()
            processed = (mask_array > threshold).astype(np.uint8)
            logger.info(f"Применен порог на основе среднего значения: {threshold:.3f}")
    
    # Проверка после бинаризации
    after_threshold_nonzero = np.sum(processed > 0)
    logger.info(f"После пороговой обработки: ненулевых вокселей={after_threshold_nonzero}")
    
    # Улучшенная морфологическая очистка
    try:
        from scipy import ndimage
        
        # Если это 3D маска, применяем 3D морфологические операции
        if len(processed.shape) == 3:
            processed = _postprocess_3d_mask(processed, min_object_size=min_object_size)
        else:
            processed = _postprocess_2d_mask(processed, min_object_size=min_object_size)
            
    except ImportError:
        logger.warning("scipy не доступен, морфологическая очистка не выполнена")
    except Exception as e:
        logger.warning(f"Ошибка при морфологической очистке: {str(e)}")
    
    # Выравнивание размеров, если указана целевая форма
    if target_shape is not None and processed.shape != target_shape:
        logger.info(f"Выравнивание размеров маски: с {processed.shape} до {target_shape}")
        processed = _resize_mask(processed, target_shape)
    
    # Финальная проверка
    final_nonzero = np.sum(processed > 0)
    logger.info(f"Финальная маска: форма={processed.shape}, ненулевых вокселей={final_nonzero} "
                f"(изначально: {initial_nonzero}, форма: {initial_shape})")
    
    # Если маска стала пустой после постобработки, предупреждаем
    if final_nonzero == 0 and initial_nonzero > 0:
        logger.warning("Внимание: маска стала пустой после постобработки! "
                     "Возможно, все объекты были отфильтрованы как слишком маленькие.")
    
    return processed


def _resize_mask(mask_array: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Изменение размера маски с сохранением бинарных значений.
    
    Args:
        mask_array: Исходная маска
        target_shape: Целевая форма
        
    Returns:
        Маска с целевыми размерами
    """
    try:
        from scipy import ndimage
        
        # Если размеры уже совпадают, возвращаем маску без изменений
        if mask_array.shape == target_shape:
            return mask_array
        
        # Обработка изменения размерности
        if len(mask_array.shape) != len(target_shape):
            # Если размерности не совпадают, создаем новую маску нужной размерности
            if len(mask_array.shape) < len(target_shape):
                # Добавляем измерения
                new_shape = [1] * (len(target_shape) - len(mask_array.shape)) + list(mask_array.shape)
                expanded_mask = np.zeros(new_shape, dtype=mask_array.dtype)
                # Копируем данные в центр новых измерений
                slices = tuple([slice(0, s) for s in mask_array.shape])
                expanded_mask[(slice(None),) * (len(target_shape) - len(mask_array.shape)) + slices] = mask_array
                mask_array = expanded_mask
            else:
                # Уменьшаем размерность, проецируя на первые измерения
                # Берем максимальное значение по дополнительным измерениям
                while len(mask_array.shape) > len(target_shape):
                    mask_array = np.max(mask_array, axis=0)
        
        # Вычисление коэффициентов масштабирования
        zoom_factors = [target / current for target, current in zip(target_shape, mask_array.shape)]
        
        # Масштабирование маски
        resized_mask = ndimage.zoom(mask_array, zoom_factors, order=0)  # order=0 для nearest-neighbor
        
        # Бинаризация результата
        resized_mask = (resized_mask > 0.5).astype(mask_array.dtype)
        
        return resized_mask
        
    except ImportError:
        # Если scipy недоступен, используем простое обрезание/дополнение
        if mask_array.shape == target_shape:
            return mask_array
        
        result = np.zeros(target_shape, dtype=mask_array.dtype)
        
        # Вычисляем минимальные размеры для копирования
        min_dims = [min(mask_array.shape[i], target_shape[i]) for i in range(min(len(mask_array.shape), len(target_shape)))]
        
        # Копируем данные с учетом размерности
        if len(mask_array.shape) == 3 and len(target_shape) == 3:
            result[:min_dims[0], :min_dims[1], :min_dims[2]] = mask_array[:min_dims[0], :min_dims[1], :min_dims[2]]
        elif len(mask_array.shape) == 2 and len(target_shape) == 2:
            result[:min_dims[0], :min_dims[1]] = mask_array[:min_dims[0], :min_dims[1]]
        elif len(mask_array.shape) == 2 and len(target_shape) == 3:
            center_slice = target_shape[0] // 2
            result[center_slice, :min_dims[0], :min_dims[1]] = mask_array[:min_dims[0], :min_dims[1]]
        elif len(mask_array.shape) == 3 and len(target_shape) == 2:
            center_slice = mask_array.shape[0] // 2
            result[:min_dims[0], :min_dims[1]] = mask_array[center_slice, :min_dims[0], :min_dims[1]]
        
        return result


def _postprocess_3d_mask(mask_3d: np.ndarray, min_object_size: Optional[int] = None) -> np.ndarray:
    """
    Улучшенная постобработка 3D маски с использованием морфологических операций и фильтрацией по размеру.
    
    Args:
        mask_3d: 3D маска сегментации
        min_object_size: Минимальный размер объекта в вокселях. Если None, вычисляется автоматически
            как 1% от общего числа вокселей маски, но не менее 100.
    
    Returns:
        Постобработанная 3D маска
    """
    from scipy import ndimage
    
    # Автоматическое определение минимального размера объекта
    if min_object_size is None:
        total_voxels = mask_3d.size
        min_object_size = max(100, int(total_voxels * 0.001))  # 0.1% от общего объема, но минимум 100
        logger.info(f"Автоматически установлен min_object_size={min_object_size} для 3D маски")
    
    # Удаляем маленькие объекты (шумы)
    # Универсальный подход: вызываем label без return_num, затем вычисляем количество объектов
    labeled_array = ndimage.label(mask_3d)
    # Если ndimage.label возвращает кортеж (массив, количество) - обработаем
    if isinstance(labeled_array, tuple):
        labeled_array, num_features = labeled_array
    else:
        # Иначе это только массив меток
        if hasattr(labeled_array, 'max'):
            num_features = np.max(labeled_array)
        else:
            num_features = 0
    
    if num_features > 0:
        # Вычисляем размеры каждого объекта
        object_sizes = ndimage.sum(mask_3d, labeled_array, range(1, num_features + 1))
        
        # Логирование информации об объектах
        logger.info(f"Найдено {num_features} 3D объектов, размеры: {object_sizes}")
        
        # Фильтрация объектов по минимальному размеру
        valid_objects = []
        for i, size in enumerate(object_sizes):
            if size >= min_object_size:
                valid_objects.append((i + 1, size))  # i+1 соответствует метке
        
        if not valid_objects:
            logger.warning(f"Все объекты меньше min_object_size={min_object_size}, возвращаем пустую маску")
            return np.zeros_like(mask_3d, dtype=np.uint8)
        
        # Выбираем самый большой объект из валидных
        valid_objects.sort(key=lambda x: x[1], reverse=True)
        largest_object_label, largest_size = valid_objects[0]
        
        # Создаем маску только с самым большим объектом
        new_mask = np.zeros_like(mask_3d)
        new_mask[labeled_array == largest_object_label] = 1
        
        # Морфологические операции для удаления шума и сглаживания
        # Закрытие для заполнения дыр (используем структурный элемент 3x3x3)
        new_mask = ndimage.binary_closing(new_mask, structure=np.ones((3, 3, 3)))

        # Открытие для удаления мелких выступов (структурный элемент 3x3x3)
        new_mask = ndimage.binary_opening(new_mask, structure=np.ones((3, 3, 3)))
        
        # Удаление возможных оставшихся мелких объектов после морфологических операций
        labeled_result = ndimage.label(new_mask)
        if isinstance(labeled_result, tuple):
            labeled_processed, num_processed = labeled_result
        else:
            labeled_processed = labeled_result
            if hasattr(labeled_processed, 'max'):
                num_processed = np.max(labeled_processed)
            else:
                num_processed = 0
        
        if num_processed > 1:
            sizes = ndimage.sum(new_mask, labeled_processed, range(1, num_processed + 1))
            largest_label = np.argmax(sizes) + 1
            new_mask = (labeled_processed == largest_label)
        
        processed = np.asarray(new_mask).astype(np.uint8)
        logger.info(f"3D морфологическая очистка: сохранен объект размером {largest_size} вокселей "
                    f"(из {len(valid_objects)} валидных объектов)")
        
        return processed
    
    return mask_3d


def _postprocess_2d_mask(mask_2d: np.ndarray, min_object_size: Optional[int] = None) -> np.ndarray:
    """
    Улучшенная постобработка 2D маски с использованием морфологических операций и фильтрацией по размеру.
    
    Args:
        mask_2d: 2D маска сегментации
        min_object_size: Минимальный размер объекта в пикселях. Если None, вычисляется автоматически
            как 0.5% от общего числа пикселей маски, но не менее 10.
    
    Returns:
        Постобработанная 2D маска
    """
    from scipy import ndimage
    
    # Автоматическое определение минимального размера объекта
    if min_object_size is None:
        total_pixels = mask_2d.size
        min_object_size = max(10, int(total_pixels * 0.005))  # 0.5% от общего объема, но минимум 10
        logger.info(f"Автоматически установлен min_object_size={min_object_size} для 2D маски")
    
    # Удаляем маленькие объекты (шумы)
    # Универсальный подход: вызываем label без return_num, затем вычисляем количество объектов
    labeled_array = ndimage.label(mask_2d)
    # Если ndimage.label возвращает кортеж (массив, количество) - обработаем
    if isinstance(labeled_array, tuple):
        labeled_array, num_features = labeled_array
    else:
        # Иначе это только массив меток
        if hasattr(labeled_array, 'max'):
            num_features = np.max(labeled_array)
        else:
            num_features = 0
    
    if num_features > 0:
        # Вычисляем размеры каждого объекта
        object_sizes = ndimage.sum(mask_2d, labeled_array, range(1, num_features + 1))
        
        # Логирование информации об объектах
        logger.info(f"Найдено {num_features} 2D объектов, размеры: {object_sizes}")
        
        # Фильтрация объектов по минимальному размеру
        valid_objects = []
        for i, size in enumerate(object_sizes):
            if size >= min_object_size:
                valid_objects.append((i + 1, size))  # i+1 соответствует метке
        
        if not valid_objects:
            logger.warning(f"Все объекты меньше min_object_size={min_object_size}, возвращаем пустую маску")
            return np.zeros_like(mask_2d, dtype=np.uint8)
        
        # Выбираем самый большой объект из валидных
        valid_objects.sort(key=lambda x: x[1], reverse=True)
        largest_object_label, largest_size = valid_objects[0]
        
        # Создаем маску только с самым большим объектом
        new_mask = np.zeros_like(mask_2d)
        new_mask[labeled_array == largest_object_label] = 1
        
        # Морфологические операции для удаления шума и сглаживания
        # Закрытие для заполнения дыр (используем структурный элемент 3x3)
        new_mask = ndimage.binary_closing(new_mask, structure=np.ones((3, 3)))

        # Открытие для удаления мелких выступов (структурный элемент 3x3)
        new_mask = ndimage.binary_opening(new_mask, structure=np.ones((3, 3)))
        
        # Удаление возможных оставшихся мелких объектов после морфологических операций
        labeled_result = ndimage.label(new_mask)
        if isinstance(labeled_result, tuple):
            labeled_processed, num_processed = labeled_result
        else:
            labeled_processed = labeled_result
            if hasattr(labeled_processed, 'max'):
                num_processed = np.max(labeled_processed)
            else:
                num_processed = 0
        
        if num_processed > 1:
            sizes = ndimage.sum(new_mask, labeled_processed, range(1, num_processed + 1))
            largest_label = np.argmax(sizes) + 1
            new_mask = (labeled_processed == largest_label)
        
        processed = np.asarray(new_mask).astype(np.uint8)
        logger.info(f"2D морфологическая очистка: сохранен объект размером {largest_size} пикселей "
                    f"(из {len(valid_objects)} валидных объектов)")
        
        return processed
    
    return mask_2d


def validate_dicom_folder(folder_path: str) -> bool:
    """
    Проверить, содержит ли папка DICOM файлы.
    
    Args:
        folder_path (str): Путь к папке для проверки
        
    Returns:
        bool: True, если папка содержит DICOM файлы, иначе False
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return False
        
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
        return len(series_ids) > 0
    except:
        return False


def get_dicom_files_count(folder_path: str) -> Dict[str, int]:
    """
    Получить количество DICOM файлов в каждой серии.
    
    Args:
        folder_path (str): Путь к папке с DICOM файлами
        
    Returns:
        Dict[str, int]: Словарь с ID серий и количеством файлов в каждой
    """
    result = {}
    
    if not validate_dicom_folder(folder_path):
        return result
        
    try:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(folder_path)
        for series_id in series_ids:
            file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(folder_path, series_id)
            result[series_id] = len(file_names)
    except Exception as e:
        logger.error(f"Ошибка при подсчете DICOM файлов: {str(e)}")
        
    return result


def downsample_image(image: np.ndarray, target_size: Tuple[int, int], method: str = 'bilinear') -> np.ndarray:
    """
    Даунсэмплировать изображение до заданного размера.
    
    Args:
        image: Входное изображение
        target_size: Целевой размер (height, width)
        method: Метод интерполяции ('bilinear', 'nearest', 'lanczos')
        
    Returns:
        Даунсэмплированное изображение
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    if (h, w) == (target_h, target_w):
        return image
    
    try:
        import cv2
        
        # Определяем метод интерполяции
        if method == 'bilinear':
            interpolation = cv2.INTER_LINEAR
        elif method == 'nearest':
            interpolation = cv2.INTER_NEAREST
        elif method == 'lanczos':
            interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_LINEAR  # по умолчанию
            
        # Применяем resize
        downsampled = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        return downsampled
        
    except ImportError:
        # Если OpenCV недоступен, используем scipy
        from scipy import ndimage
        import numpy as np
        
        # Вычисляем коэффициенты масштабирования
        scale_h = target_h / h
        scale_w = target_w / w
        
        if len(image.shape) == 3:  # цветное изображение
            zoom_factors = (scale_h, scale_w, 1)
        else:  # градации серого
            zoom_factors = (scale_h, scale_w)
            
        downsampled = ndimage.zoom(image, zoom_factors, order=1)
        return downsampled


def load_dicom_series(dicom_folder_path: str, series_id: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Загрузить DICOM серию и вернуть словарь с изображениями.
    
    Args:
        dicom_folder_path: Путь к папке с DICOM файлами
        series_id: ID DICOM серии (опционально, если None - используется первая серия)
        
    Returns:
        Словарь с изображениями (ключ - имя файла или номер среза)
    """
    dicom_loader = DicomLoader()
    
    # Получаем список серий в папке
    series_ids = dicom_loader.get_dicom_series(dicom_folder_path)
    
    if not series_ids:
        raise ValueError(f"Не найдено DICOM серий в папке {dicom_folder_path}")
    
    # Если series_id не указан, используем первую серию
    if series_id is None:
        series_id = series_ids[0]
    elif series_id not in series_ids:
        raise ValueError(f"Серия с ID {series_id} не найдена в папке {dicom_folder_path}. Доступные серии: {series_ids}")
    
    sitk_image = dicom_loader.load_series(dicom_folder_path, series_id)
    image_array = dicom_loader.convert_to_numpy(sitk_image)
    
    # Создаем словарь с изображениями
    images_dict = {}
    
    if len(image_array.shape) == 3:
        # 3D серия - каждый срез как отдельное изображение
        for i in range(image_array.shape[0]):
            images_dict[f"slice_{i:03d}"] = image_array[i]
    else:
        # 2D изображение
        images_dict["image"] = image_array
    
    return images_dict


class MaskedVolumeProcessor:
    """
    Класс для обработки маскирования 3D объемов.
    """
    
    def __init__(self):
        self.masked_volumes_cache = {}  # Кэш для маскированных объемов
        
    def create_masked_volume(self, volume: np.ndarray, mask: np.ndarray, 
                           mode: str = 'multiply', background_value: float = 0) -> np.ndarray:
        """
        Создает маскированный 3D объем.
        
        Args:
            volume: 3D массив - оригинальный объем
            mask: 3D массив - бинарная маска
            mode: Режим маскирования ('multiply', 'overlay', 'contour', 'transparent')
            background_value: Значение для фона
            
        Returns:
            3D массив с примененной маской
        """
        cache_key = (id(volume), id(mask), mode, background_value)
        if cache_key in self.masked_volumes_cache:
            return self.masked_volumes_cache[cache_key]
            
        if mode == 'multiply':
            result = apply_mask_to_volume(volume, mask, background_value)
        elif mode == 'overlay':
            result = self._create_overlay_volume(volume, mask, background_value)
        elif mode == 'contour':
            result = self._create_contour_volume(volume, mask)
        elif mode == 'transparent':
            result = self._create_transparent_volume(volume, mask)
        else:
            raise ValueError(f"Неподдерживаемый режим маскирования: {mode}")
            
        # Кэшируем результат
        self.masked_volumes_cache[cache_key] = result
        return result
    
    def _create_overlay_volume(self, volume: np.ndarray, mask: np.ndarray, background_value: float = 0) -> np.ndarray:
        """Создает объем с наложением маски."""
        # Для каждого среза применяем маскирование
        result = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            result[i] = create_masked_slice(volume[i], mask[i], alpha=0.7, background_value=background_value)
        return result
    
    def _create_contour_volume(self, volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Создает объем с контурной обводкой."""
        # Для каждого среза применяем контурную обводку
        result = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            result[i] = create_contour_overlay(volume[i], mask[i])
        return result
    
    def _create_transparent_volume(self, volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Создает объем с полупрозрачным наложением."""
        # Для каждого среза применяем полупрозрачное наложение
        result = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            result[i] = create_transparent_overlay(volume[i], mask[i], alpha=0.5)
        return result
    
    def clear_cache(self):
        """Очищает кэш маскированных объемов."""
        self.masked_volumes_cache.clear()


def create_masked_slice_from_volume(volume: np.ndarray, mask: np.ndarray, slice_idx: int, 
                                  mode: str = 'multiply', alpha: float = 0.7, 
                                  overlay_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    Создает маскированный слайс из 3D объема.
    
    Args:
        volume: 3D массив - оригинальный объем
        mask: 3D массив - бинарная маска
        slice_idx: Индекс слайса
        mode: Режим маскирования ('multiply', 'overlay', 'contour', 'transparent')
        alpha: Прозрачность для режимов overlay и transparent
        overlay_color: Цвет для прозрачного наложения
        
    Returns:
        2D массив - маскированный слайс
    """
    if slice_idx >= volume.shape[0] or slice_idx < 0:
        raise IndexError(f"Индекс слайса {slice_idx} вне диапазона [0, {volume.shape[0]-1}]")
    
    slice_img = volume[slice_idx]
    mask_slice = mask[slice_idx]
    
    if mode == 'multiply':
        return create_masked_slice(slice_img, mask_slice, alpha=alpha)
    elif mode == 'overlay':
        return create_masked_slice(slice_img, mask_slice, alpha=alpha)
    elif mode == 'contour':
        return create_contour_overlay(slice_img, mask_slice)
    elif mode == 'transparent':
        return create_transparent_overlay(slice_img, mask_slice, alpha=alpha, overlay_color=overlay_color)
    else:
        raise ValueError(f"Неподдерживаемый режим маскирования: {mode}")