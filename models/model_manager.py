"""
Менеджер моделей для сегментации печени.

Этот модуль содержит класс ModelManager, который обеспечивает унифицированный интерфейс
для загрузки, инициализации и выполнения инференса с использованием различных моделей
сегментации печени.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod
import time
from functools import lru_cache

# Импорт монитора производительности
try:
    from ..utils.performance_monitor import performance_monitor, performance_timer
except ImportError:
    # Для локального тестирования
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
    from performance_monitor import performance_monitor, performance_timer

# Импорты моделей
try:
    # Абсолютные импорты от корня проекта
    from InferenceAnd3D.models.models import (
        get_model, BaseSegmentationModel, UNet, NnUNet, AttentionUNet,
        UNet3D, ResUNet, SwinUNETR, UNetPlusPlus, EfficientNetUNet,
        TransUNet, CoTr
    )
    # Импорт конфигурации
    from InferenceAnd3D.config import YOLO_CONFIG
except ImportError:
    try:
        # Альтернативные абсолютные импорты (если InferenceAnd3D уже в sys.path)
        from models.models import (
            get_model, BaseSegmentationModel, UNet, NnUNet, AttentionUNet,
            UNet3D, ResUNet, SwinUNETR, UNetPlusPlus, EfficientNetUNet,
            TransUNet, CoTr
        )
        # Импорт конфигурации
        from config import YOLO_CONFIG
    except ImportError:
        # Относительные импорты как запасной вариант
        from .models import (
            get_model, BaseSegmentationModel, UNet, NnUNet, AttentionUNet,
            UNet3D, ResUNet, SwinUNETR, UNetPlusPlus, EfficientNetUNet,
            TransUNet, CoTr
        )
        from ..config import YOLO_CONFIG

# Настройка логирования с поддержкой кириллицы и Unicode
logger = logging.getLogger(__name__)


class ModelSegmentator(ABC):
    """
    Абстрактный базовый класс для всех сегментаторов.
    Определяет общий интерфейс для выполнения инференса.
    """
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Выполнение инференса на изображении.
        
        Args:
            image: Входное изображение
            
        Returns:
            Маска сегментации
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """
        Загрузка весов модели.
        
        Args:
            model_path: Путь к файлу с весами модели
        """
        pass


class YOLOLiverSegmentator(ModelSegmentator):
    """
    Сегментатор на основе YOLO для сегментации печени.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Инициализация YOLO сегментатора.
        
        Args:
            device: Устройство для вычислений
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = "YOLOv11"
        
    def load_model(self, model_path: str) -> None:
        """
        Загрузка весов YOLO модели.
        
        Args:
            model_path: Путь к файлу с весами модели
        """
        try:
            # Стандартный импорт для новых версий
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info(f"YOLO модель успешно загружена из {model_path}")
        except (ImportError, AttributeError):
            try:
                # Попытка прямого импорта через модуль
                import ultralytics
                if hasattr(ultralytics, 'YOLO'):
                    self.model = ultralytics.YOLO(model_path)
                else:
                    raise ImportError("YOLO не найден в ultralytics")
            except Exception:
                raise ImportError("Не удалось импортировать YOLO из ultralytics. Убедитесь, что пакет установлен корректно.")
    
    def normalize_hu(self, image: np.ndarray, window_center: int = 40, window_width: int = 400) -> np.ndarray:
        """
        Windowing и нормализация HU значений как при обучении.
        
        Args:
            image: Входное изображение с HU значениями
            window_center: Центр окна в HU
            window_width: Ширина окна в HU
            
        Returns:
            Нормализованное изображение в диапазоне [0, 255]
        """
        min_hu = window_center - window_width // 2
        max_hu = window_center + window_width // 2
        image = np.clip(image, min_hu, max_hu)
        image = (image - min_hu) / (max_hu - min_hu)
        return (image * 255).astype(np.uint8)
    
    @performance_timer("model.yolo.predict")
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Оптимизированное выполнение инференса с использованием YOLO.
        
        Args:
            image: Входное изображение
            
        Returns:
            Маска сегментации
        """
        if self.model is None:
            raise ValueError("Модель не загружена. Вызовите метод load_model().")
        
        # Обработка 3D объема - пакетная обработка срезов
        if len(image.shape) == 3:
            # Оптимизированная пакетная обработка для больших объемов
            if image.shape[0] > 10:  # Если срезов много, обрабатываем пакетами
                return self._predict_3d_batched(image)
            else:
                # Для небольших объемов обрабатываем последовательно
                mask_slices = []
                for i, slice_img in enumerate(image):
                    mask_slice = self._predict_2d(slice_img)
                    mask_slices.append(mask_slice)
                result = np.stack(mask_slices)
                
                # Применяем постобработку для 3D маски
                result = self._postprocess_3d_mask(result)
        else:
            # Обработка 2D изображения
            result = self._predict_2d(image)
            
            # Применяем постобработку для 2D маски
            result = self._postprocess_2d_mask(result)
        
        return result
    
    def _predict_3d_batched(self, image: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """
        Пакетная обработка 3D изображения для ускорения инференса.
        
        Args:
            image: 3D изображение
            batch_size: Размер пакета для обработки
            
        Returns:
            3D маска сегментации
        """
        num_slices = image.shape[0]
        mask_slices = []
        
        # Обработка пакетами
        for i in range(0, num_slices, batch_size):
            end_idx = min(i + batch_size, num_slices)
            batch_slices = image[i:end_idx]
            
            # Обработка каждого среза в пакете
            batch_masks = []
            for slice_img in batch_slices:
                mask_slice = self._predict_2d(slice_img)
                batch_masks.append(mask_slice)
            
            mask_slices.extend(batch_masks)
        
        return np.stack(mask_slices)
    
    def _predict_2d(self, image: np.ndarray) -> np.ndarray:
        """
        Оптимизированное выполнение инференса на 2D изображении.
        
        Args:
            image: 2D изображение
            
        Returns:
            Маска сегментации
        """
        # Проверка входных данных
        if image is None or image.size == 0:
            logger.error("Получено пустое изображение")
            return np.zeros((512, 512), dtype=np.uint8)  # Возвращаем маску по умолчанию
        
        # Оптимизированная обработка различных форматов входных данных
        original_shape = image.shape
        if len(image.shape) == 3:
            # Если это 3D изображение, берем средний срез
            if image.shape[0] == 1:  # Формат (1, H, W)
                image = image.squeeze(0)
            elif image.shape[2] == 1:  # Формат (H, W, 1)
                image = image.squeeze(2)
            else:  # Формат (H, W, C) или (D, H, W)
                # Если это 3D объем, берем центральный срез
                if image.shape[0] < image.shape[2]:  # Скорее всего (D, H, W)
                    slice_idx = image.shape[0] // 2
                    image = image[slice_idx]
                else:  # Скорее всего (H, W, C)
                    image = image[:, :, 0]  # Берем первый канал
        
        # Убедимся, что изображение 2D
        if len(image.shape) != 2:
            logger.error(f"Некорректная размерность изображения: {image.shape}")
            return np.zeros((512, 512), dtype=np.uint8)
        
        # Проверка размеров изображения
        if image.shape[0] == 0 or image.shape[1] == 0:
            logger.error(f"Изображение имеет нулевые размеры: {image.shape}")
            return np.zeros((512, 512), dtype=np.uint8)
        
        # Оптимизированная нормализация HU как при обучении
        window_center = YOLO_CONFIG['window_center']
        window_width = YOLO_CONFIG['window_width']
        image = self.normalize_hu(image, window_center, window_width)
        
        # Проверка минимального размера изображения
        if image.shape[0] < 10 or image.shape[1] < 10:
            logger.warning(f"Изображение слишком маленькое: {image.shape}, увеличиваем до минимального размера")
            # Увеличиваем изображение до минимального размера
            try:
                import cv2
                new_width = max(10, image.shape[1])
                new_height = max(10, image.shape[0])
                image = cv2.resize(image, (new_width, new_height))
            except ImportError:
                # Если cv2 недоступен, используем numpy resize
                from scipy import ndimage
                new_width = max(10, image.shape[1])
                new_height = max(10, image.shape[0])
                zoom_factor = (new_height / image.shape[0], new_width / image.shape[1])
                image = ndimage.zoom(image, zoom_factor, order=1)
        
        try:
            # Оптимизированная конвертация в 3 канала для YOLO
            if len(image.shape) == 2:
                # Используем broadcast_to вместо stack для экономии памяти
                image = np.broadcast_to(image[:, :, np.newaxis], image.shape + (3,))
            elif len(image.shape) == 3 and image.shape[-1] == 1:
                # Конвертируем [H, W, 1] в [H, W, 3]
                image = np.repeat(image, 3, axis=-1)
            
            # Выполнение инференса с низким порогом уверенности
            conf_threshold = YOLO_CONFIG['confidence_threshold']
            results = self.model(image, conf=conf_threshold)
            
            # Извлечение маски из результатов с улучшенной обработкой
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    mask = result.masks.data.cpu().numpy().squeeze()
                    if mask.ndim == 0:
                        logger.warning("YOLO вернула маску с нулевой размерностью, создаем пустую маску")
                        mask = np.zeros_like(image[:, :, 0])  # Берем первый канал для маски
                    else:
                        logger.info(f"YOLO успешно создала маску: форма={mask.shape}, ненулевых пикселей={np.sum(mask > 0)}")
                elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    # Если маски нет, но есть боксы, создаем маску на основе боксов
                    logger.warning(f"YOLO нашла {len(result.boxes)} объектов, но маски отсутствуют. Создаем маску на основе боксов.")
                    mask = self._create_mask_from_boxes(result.boxes, image.shape[:2])
                else:
                    logger.warning(f"YOLO не нашел объектов (порог: {conf_threshold}), создаем пустую маску")
                    mask = np.zeros_like(image[:, :, 0])  # Берем первый канал для маски
            else:
                logger.warning(f"YOLO вернул пустые результаты, создаем пустую маску")
                mask = np.zeros_like(image[:, :, 0])  # Берем первый канал для маски
                
            return mask
        except Exception as e:
            logger.error(f"Ошибка при выполнении инференса YOLO: {str(e)}")
            # Возвращаем пустую маску того же размера, что и входное изображение
            if len(image.shape) == 3:
                return np.zeros_like(image[:, :, 0])  # Берем первый канал для маски
            else:
                return np.zeros_like(image)
    
    def _create_mask_from_boxes(self, boxes, image_shape):
        """
        Создание маски на основе ограничивающих рамок.
        
        Args:
            boxes: Ограничивающие рамки от YOLO
            image_shape: Размер изображения (height, width)
            
        Returns:
            Маска сегментации
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        try:
            # Конвертация боксов в numpy
            if hasattr(boxes, 'xyxy'):
                # Формат xyxy (x1, y1, x2, y2)
                bboxes = boxes.xyxy.cpu().numpy()
            elif hasattr(boxes, 'xywh'):
                # Формат xywh (x, y, width, height)
                xywh = boxes.xywh.cpu().numpy()
                bboxes = np.zeros_like(xywh)
                bboxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
                bboxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
                bboxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
                bboxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2
            else:
                logger.warning("Неизвестный формат боксов")
                return mask
            
            # Заполнение маски
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.astype(int)
                # Ограничение координат в пределах изображения
                x1 = max(0, min(x1, image_shape[1] - 1))
                y1 = max(0, min(y1, image_shape[0] - 1))
                x2 = max(0, min(x2, image_shape[1] - 1))
                y2 = max(0, min(y2, image_shape[0] - 1))
                
                # Заполнение прямоугольника
                mask[y1:y2, x1:x2] = 1
                
            logger.info(f"Создана маска на основе {len(bboxes)} ограничивающих рамок")
            return mask
            
        except Exception as e:
            logger.error(f"Ошибка при создании маски из боксов: {str(e)}")
            return mask
    
    def _postprocess_2d_mask(self, mask_2d: np.ndarray) -> np.ndarray:
        """
        Постобработка 2D маски для YOLO.
        
        Args:
            mask_2d: 2D маска сегментации
            
        Returns:
            Постобработанная 2D маска
        """
        try:
            from scipy import ndimage
            
            # Удаляем маленькие объекты (шумы)
            labeled_array, num_features = ndimage.label(mask_2d)
            
            if num_features > 0:
                # Вычисляем размеры каждого объекта
                object_sizes = ndimage.sum(mask_2d, labeled_array, range(1, num_features + 1))
                
                # Для печени оставляем только самый большой объект
                if len(object_sizes) > 0:
                    largest_object_idx = np.argmax(object_sizes) + 1  # +1 т.к. метки начинаются с 1
                    
                    # Создаем маску только с самым большим объектом
                    new_mask = np.zeros_like(mask_2d)
                    new_mask[labeled_array == largest_object_idx] = 1
                    
                    # Применяем морфологическое закрытие для заполнения маленьких дыр
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((2, 2)))
                    
                    # Применяем морфологическое открытие для удаления мелких выступов
                    new_mask = ndimage.binary_opening(new_mask, structure=np.ones((2, 2)))
                    
                    # Сглаживание контура
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((5, 5)))
                    
                    processed = new_mask.astype(np.uint8)
                    logger.info(f"YOLO 2D постобработка: сохранен самый большой объект размером {object_sizes[largest_object_idx-1]} вокселей")
                    
                    # Проверка, что маска не стала полностью пустой после обработки
                    if np.sum(processed) == 0:
                        logger.warning("YOLO 2D постобработка: маска стала пустой, возвращаем исходную маску")
                        return mask_2d.astype(np.uint8)
                    
                    return processed
            
            return mask_2d
            
        except ImportError:
            logger.warning("scipy не доступен, постобработка YOLO 2D маски не выполнена")
            return mask_2d
        except Exception as e:
            logger.warning(f"Ошибка при постобработке YOLO 2D маски: {str(e)}")
            return mask_2d
    
    def _postprocess_3d_mask(self, mask_3d: np.ndarray) -> np.ndarray:
        """
        Постобработка 3D маски для YOLO.
        
        Args:
            mask_3d: 3D маска сегментации
            
        Returns:
            Постобработанная 3D маска
        """
        try:
            from scipy import ndimage
            
            # Удаляем маленькие объекты (шумы)
            labeled_array, num_features = ndimage.label(mask_3d)
            
            if num_features > 0:
                # Вычисляем размеры каждого объекта
                object_sizes = ndimage.sum(mask_3d, labeled_array, range(1, num_features + 1))
                
                # Для печени оставляем только самый большой объект
                if len(object_sizes) > 0:
                    largest_object_idx = np.argmax(object_sizes) + 1  # +1 т.к. метки начинаются с 1
                    
                    # Создаем маску только с самым большим объектом
                    new_mask = np.zeros_like(mask_3d)
                    new_mask[labeled_array == largest_object_idx] = 1
                    
                    # Применяем морфологическое закрытие для заполнения маленьких дыр
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((2, 2, 2)))
                    
                    # Применяем морфологическое открытие для удаления мелких выступов
                    new_mask = ndimage.binary_opening(new_mask, structure=np.ones((2, 2, 2)))
                    
                    # Сглаживание поверхности
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((5, 5, 5)))
                    
                    processed = new_mask.astype(np.uint8)
                    logger.info(f"YOLO 3D постобработка: сохранен самый большой объект размером {object_sizes[largest_object_idx-1]} вокселей")
                    
                    # Проверка, что маска не стала полностью пустой после обработки
                    if np.sum(processed) == 0:
                        logger.warning("YOLO 3D постобработка: маска стала пустой, возвращаем исходную маску")
                        return mask_3d.astype(np.uint8)
                    
                    return processed
            
            return mask_3d
            
        except ImportError:
            logger.warning("scipy не доступен, постобработка YOLO 3D маски не выполнена")
            return mask_3d
        except Exception as e:
            logger.warning(f"Ошибка при постобработке YOLO 3D маски: {str(e)}")
            return mask_3d


class UNetLiverSegmentator(ModelSegmentator):
    """
    Сегментатор на основе U-Net для сегментации печени.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Инициализация U-Net сегментатора.
        
        Args:
            device: Устройство для вычислений
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = "U-Net"
        
    def load_model(self, model_path: str) -> None:
        """
        Загрузка весов U-Net модели.
        
        Args:
            model_path: Путь к файлу с весами модели
        """
        try:
            # Создание модели
            self.model = get_model('unet', n_channels=1, n_classes=1)
            
            # Загрузка весов
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Проверка формата чекпоинта
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"U-Net модель успешно загружена из {model_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке U-Net модели: {str(e)}")
            raise
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Оптимизированное выполнение инференса с использованием U-Net.
        
        Args:
            image: Входное изображение
            
        Returns:
            Маска сегментации
        """
        start_time = time.time()
        
        if self.model is None:
            raise ValueError("Модель не загружена. Вызовите метод load_model().")
        
        # Обработка 3D объема - пакетная обработка срезов
        if len(image.shape) == 3:
            # Оптимизированная пакетная обработка для больших объемов
            if image.shape[0] > 10:  # Если срезов много, обрабатываем пакетами
                result = self._predict_3d_batched(image, batch_size=8)
            else:
                # Для небольших объемов обрабатываем последовательно
                mask_slices = []
                for i, slice_img in enumerate(image):
                    mask_slice = self._predict_2d(slice_img)
                    mask_slices.append(mask_slice)
                result = np.stack(mask_slices)
                
                # Применяем постобработку для 3D маски
                result = self._postprocess_3d_mask(result)
        else:
            # Обработка 2D изображения
            result = self._predict_2d(image)
            
            # Применяем постобработку для 2D маски
            result = self._postprocess_2d_mask(result)
        
        processing_time = time.time() - start_time
        logger.info(f"U-Net инференс выполнен за {processing_time:.3f}с для изображения формы {image.shape}")
        return result
    
    def _predict_3d_batched(self, image: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """
        Пакетная обработка 3D изображения для ускорения инференса.
        
        Args:
            image: 3D изображение
            batch_size: Размер пакета для обработки
            
        Returns:
            3D маска сегментации
        """
        num_slices = image.shape[0]
        mask_slices = []
        
        # Обработка пакетами
        for i in range(0, num_slices, batch_size):
            end_idx = min(i + batch_size, num_slices)
            batch_slices = image[i:end_idx]
            
            # Создание тензора для пакета
            batch_tensors = []
            for slice_img in batch_slices:
                # Конвертация в тензор
                if len(slice_img.shape) == 2:
                    slice_img = np.expand_dims(slice_img, axis=0)
                if len(slice_img.shape) == 3:
                    slice_img = np.expand_dims(slice_img, axis=0)
                
                tensor_img = torch.from_numpy(slice_img).float().to(self.device)
                batch_tensors.append(tensor_img)
            
            # Объединение в пакетный тензор
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Выполнение инференса для всего пакета
            with torch.no_grad():
                batch_output = self.model(batch_tensor)
            
            # Применение сигмоиды и порога для всего пакета
            batch_masks = torch.sigmoid(batch_output).squeeze().cpu().numpy()
            if batch_masks.ndim == 1:
                batch_masks = batch_masks[np.newaxis, ...]
            
            batch_masks = (batch_masks > 0.5).astype(np.uint8)
            
            # Добавление результатов
            for mask in batch_masks:
                mask_slices.append(mask)
        
        return np.stack(mask_slices)
    
    def _predict_2d(self, image: np.ndarray) -> np.ndarray:
        """
        Оптимизированное выполнение инференса на 2D изображении.
        
        Args:
            image: 2D изображение
            
        Returns:
            Маска сегментации
        """
        # Оптимизированная конвертация в тензор
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Оптимизированная конвертация в тензор с правильным типом данных
        tensor_img = torch.from_numpy(image).float().to(self.device, non_blocking=True)
        
        # Выполнение инференса
        with torch.no_grad():
            output = self.model(tensor_img)
            
        # Оптимизированное применение сигмоиды и порога
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Убедимся, что маска 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        # Применение порога с сохранением типа uint8
        mask = (mask > 0.5).astype(np.uint8)
        
        # Логирование информации о маске
        logger.info(f"U-Net создала маску: форма={mask.shape}, ненулевых пикселей={np.sum(mask > 0)}")
        
        return mask
    
    def _postprocess_2d_mask(self, mask_2d: np.ndarray) -> np.ndarray:
        """
        Постобработка 2D маски для U-Net.
        
        Args:
            mask_2d: 2D маска сегментации
            
        Returns:
            Постобработанная 2D маска
        """
        try:
            from scipy import ndimage
            
            # Удаляем маленькие объекты (шумы)
            labeled_array, num_features = ndimage.label(mask_2d)
            
            if num_features > 0:
                # Вычисляем размеры каждого объекта
                object_sizes = ndimage.sum(mask_2d, labeled_array, range(1, num_features + 1))
                
                # Для печени оставляем только самый большой объект
                if len(object_sizes) > 0:
                    largest_object_idx = np.argmax(object_sizes) + 1  # +1 т.к. метки начинаются с 1
                    
                    # Создаем маску только с самым большим объектом
                    new_mask = np.zeros_like(mask_2d)
                    new_mask[labeled_array == largest_object_idx] = 1
                    
                    # Применяем морфологическое закрытие для заполнения маленьких дыр
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((3, 3)))
                    
                    # Применяем морфологическое открытие для удаления мелких выступов
                    new_mask = ndimage.binary_opening(new_mask, structure=np.ones((3, 3)))
                    
                    # Сглаживание контура
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((5, 5)))
                    
                    processed = new_mask.astype(np.uint8)
                    logger.info(f"U-Net 2D постобработка: сохранен самый большой объект размером {object_sizes[largest_object_idx-1]} вокселей")
                    
                    return processed
            
            return mask_2d
            
        except ImportError:
            logger.warning("scipy не доступен, постобработка U-Net 2D маски не выполнена")
            return mask_2d
        except Exception as e:
            logger.warning(f"Ошибка при постобработке U-Net 2D маски: {str(e)}")
            return mask_2d
    
    def _postprocess_3d_mask(self, mask_3d: np.ndarray) -> np.ndarray:
        """
        Постобработка 3D маски для U-Net.
        
        Args:
            mask_3d: 3D маска сегментации
            
        Returns:
            Постобработанная 3D маска
        """
        try:
            from scipy import ndimage
            
            # Удаляем маленькие объекты (шумы)
            labeled_array, num_features = ndimage.label(mask_3d)
            
            if num_features > 0:
                # Вычисляем размеры каждого объекта
                object_sizes = ndimage.sum(mask_3d, labeled_array, range(1, num_features + 1))
                
                # Для печени оставляем только самый большой объект
                if len(object_sizes) > 0:
                    largest_object_idx = np.argmax(object_sizes) + 1  # +1 т.к. метки начинаются с 1
                    
                    # Создаем маску только с самым большим объектом
                    new_mask = np.zeros_like(mask_3d)
                    new_mask[labeled_array == largest_object_idx] = 1
                    
                    # Применяем морфологическое закрытие для заполнения маленьких дыр
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((3, 3, 3)))
                    
                    # Применяем морфологическое открытие для удаления мелких выступов
                    new_mask = ndimage.binary_opening(new_mask, structure=np.ones((3, 3, 3)))
                    
                    # Сглаживание поверхности
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((5, 5, 5)))
                    
                    processed = new_mask.astype(np.uint8)
                    logger.info(f"U-Net 3D постобработка: сохранен самый большой объект размером {object_sizes[largest_object_idx-1]} вокселей")
                    
                    return processed
            
            return mask_3d
            
        except ImportError:
            logger.warning("scipy не доступен, постобработка U-Net 3D маски не выполнена")
            return mask_3d
        except Exception as e:
            logger.warning(f"Ошибка при постобработке U-Net 3D маски: {str(e)}")
            return mask_3d


class NNUNetLiverSegmentator(ModelSegmentator):
    """
    Сегментатор на основе nnU-Net для сегментации печени.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Инициализация nnU-Net сегментатора.
        
        Args:
            device: Устройство для вычислений
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = "nnU-Net"
        
    def load_model(self, model_path: str) -> None:
        """
        Загрузка весов nnU-Net модели.
        
        Args:
            model_path: Путь к файлу с весами модели
        """
        try:
            # Создание модели
            self.model = get_model('nnunet', n_channels=1, n_classes=1)
            
            # Загрузка весов
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Проверка формата чекпоинта
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"nnU-Net модель успешно загружена из {model_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке nnU-Net модели: {str(e)}")
            raise
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Оптимизированное выполнение инференса с использованием nnU-Net.
        
        Args:
            image: Входное изображение
            
        Returns:
            Маска сегментации
        """
        start_time = time.time()
        
        if self.model is None:
            raise ValueError("Модель не загружена. Вызовите метод load_model().")
        
        # Обработка 3D объема - пакетная обработка срезов
        if len(image.shape) == 3:
            # Оптимизированная пакетная обработка для больших объемов
            if image.shape[0] > 10:  # Если срезов много, обрабатываем пакетами
                result = self._predict_3d_batched(image, batch_size=8)
            else:
                # Для небольших объемов обрабатываем последовательно
                mask_slices = []
                for i, slice_img in enumerate(image):
                    mask_slice = self._predict_2d(slice_img)
                    mask_slices.append(mask_slice)
                result = np.stack(mask_slices)
                
                # Применяем постобработку для 3D маски
                result = self._postprocess_3d_mask(result)
        else:
            # Обработка 2D изображения
            result = self._predict_2d(image)
            
            # Применяем постобработку для 2D маски
            result = self._postprocess_2d_mask(result)
        
        processing_time = time.time() - start_time
        logger.info(f"nnU-Net инференс выполнен за {processing_time:.3f}с для изображения формы {image.shape}")
        return result
    
    def _predict_3d_batched(self, image: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """
        Пакетная обработка 3D изображения для ускорения инференса.
        
        Args:
            image: 3D изображение
            batch_size: Размер пакета для обработки
            
        Returns:
            3D маска сегментации
        """
        num_slices = image.shape[0]
        mask_slices = []
        
        # Обработка пакетами
        for i in range(0, num_slices, batch_size):
            end_idx = min(i + batch_size, num_slices)
            batch_slices = image[i:end_idx]
            
            # Создание тензора для пакета
            batch_tensors = []
            for slice_img in batch_slices:
                # Конвертация в тензор
                if len(slice_img.shape) == 2:
                    slice_img = np.expand_dims(slice_img, axis=0)
                if len(slice_img.shape) == 3:
                    slice_img = np.expand_dims(slice_img, axis=0)
                
                tensor_img = torch.from_numpy(slice_img).float().to(self.device)
                batch_tensors.append(tensor_img)
            
            # Объединение в пакетный тензор
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Выполнение инференса для всего пакета
            with torch.no_grad():
                batch_output = self.model(batch_tensor)
            
            # Применение сигмоиды и порога для всего пакета
            batch_masks = torch.sigmoid(batch_output).squeeze().cpu().numpy()
            if batch_masks.ndim == 1:
                batch_masks = batch_masks[np.newaxis, ...]
            
            batch_masks = (batch_masks > 0.5).astype(np.uint8)
            
            # Добавление результатов
            for mask in batch_masks:
                mask_slices.append(mask)
        
        return np.stack(mask_slices)
    
    def _predict_2d(self, image: np.ndarray) -> np.ndarray:
        """
        Оптимизированное выполнение инференса на 2D изображении.
        
        Args:
            image: 2D изображение
            
        Returns:
            Маска сегментации
        """
        # Оптимизированная конвертация в тензор
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Оптимизированная конвертация в тензор с правильным типом данных
        tensor_img = torch.from_numpy(image).float().to(self.device, non_blocking=True)
        
        # Выполнение инференса
        with torch.no_grad():
            output = self.model(tensor_img)
            
        # Оптимизированное применение сигмоиды и порога
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Убедимся, что маска 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        # Применение порога с сохранением типа uint8
        mask = (mask > 0.5).astype(np.uint8)
        
        # Логирование информации о маске
        logger.info(f"nnU-Net создала маску: форма={mask.shape}, ненулевых пикселей={np.sum(mask > 0)}")
        
        return mask
    
    def _postprocess_2d_mask(self, mask_2d: np.ndarray) -> np.ndarray:
        """
        Постобработка 2D маски для nnU-Net.
        
        Args:
            mask_2d: 2D маска сегментации
            
        Returns:
            Постобработанная 2D маска
        """
        try:
            from scipy import ndimage
            
            # Удаляем маленькие объекты (шумы)
            labeled_array, num_features = ndimage.label(mask_2d)
            
            if num_features > 0:
                # Вычисляем размеры каждого объекта
                object_sizes = ndimage.sum(mask_2d, labeled_array, range(1, num_features + 1))
                
                # Для печени оставляем только самый большой объект
                if len(object_sizes) > 0:
                    largest_object_idx = np.argmax(object_sizes) + 1  # +1 т.к. метки начинаются с 1
                    
                    # Создаем маску только с самым большим объектом
                    new_mask = np.zeros_like(mask_2d)
                    new_mask[labeled_array == largest_object_idx] = 1
                    
                    # Применяем морфологическое закрытие для заполнения маленьких дыр
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((3, 3)))
                    
                    # Применяем морфологическое открытие для удаления мелких выступов
                    new_mask = ndimage.binary_opening(new_mask, structure=np.ones((3, 3)))
                    
                    # Сглаживание контура
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((5, 5)))
                    
                    processed = new_mask.astype(np.uint8)
                    logger.info(f"nnU-Net 2D постобработка: сохранен самый большой объект размером {object_sizes[largest_object_idx-1]} вокселей")
                    
                    return processed
            
            return mask_2d
            
        except ImportError:
            logger.warning("scipy не доступен, постобработка nnU-Net 2D маски не выполнена")
            return mask_2d
        except Exception as e:
            logger.warning(f"Ошибка при постобработке nnU-Net 2D маски: {str(e)}")
            return mask_2d
    
    def _postprocess_3d_mask(self, mask_3d: np.ndarray) -> np.ndarray:
        """
        Постобработка 3D маски для nnU-Net.
        
        Args:
            mask_3d: 3D маска сегментации
            
        Returns:
            Постобработанная 3D маска
        """
        try:
            from scipy import ndimage
            
            # Удаляем маленькие объекты (шумы)
            labeled_array, num_features = ndimage.label(mask_3d)
            
            if num_features > 0:
                # Вычисляем размеры каждого объекта
                object_sizes = ndimage.sum(mask_3d, labeled_array, range(1, num_features + 1))
                
                # Для печени оставляем только самый большой объект
                if len(object_sizes) > 0:
                    largest_object_idx = np.argmax(object_sizes) + 1  # +1 т.к. метки начинаются с 1
                    
                    # Создаем маску только с самым большим объектом
                    new_mask = np.zeros_like(mask_3d)
                    new_mask[labeled_array == largest_object_idx] = 1
                    
                    # Применяем морфологическое закрытие для заполнения маленьких дыр
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((3, 3, 3)))
                    
                    # Применяем морфологическое открытие для удаления мелких выступов
                    new_mask = ndimage.binary_opening(new_mask, structure=np.ones((3, 3, 3)))
                    
                    # Сглаживание поверхности
                    new_mask = ndimage.binary_closing(new_mask, structure=np.ones((5, 5, 5)))
                    
                    processed = new_mask.astype(np.uint8)
                    logger.info(f"nnU-Net 3D постобработка: сохранен самый большой объект размером {object_sizes[largest_object_idx-1]} вокселей")
                    
                    return processed
            
            return mask_3d
            
        except ImportError:
            logger.warning("scipy не доступен, постобработка nnU-Net 3D маски не выполнена")
            return mask_3d
        except Exception as e:
            logger.warning(f"Ошибка при постобработке nnU-Net 3D маски: {str(e)}")
            return mask_3d


class ModelManager:
    """
    Менеджер моделей для управления различными моделями сегментации печени.
    
    Обеспечивает унифицированный интерфейс для загрузки, кэширования и выполнения
    инференса с использованием различных моделей сегментации.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Инициализация менеджера моделей.
        
        Args:
            device: Устройство для вычислений
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models: Dict[str, ModelSegmentator] = {}
        self.model_paths: Dict[str, str] = {}
        
        # Регистрация доступных моделей
        self._register_models()
        
        logger.info(f"ModelManager инициализирован на устройстве: {self.device}")
    
    def _register_models(self):
        """Регистрация доступных моделей."""
        self.available_models = {
            'YOLOv11': YOLOLiverSegmentator,
            'U-Net': UNetLiverSegmentator,
            'nnU-Net': NNUNetLiverSegmentator,
        }
        
        logger.info(f"Зарегистрировано моделей: {list(self.available_models.keys())}")
    
    def load_model(self, model_name: str, model_path: str) -> bool:
        """
        Загрузка модели с указанным именем и путем к весам.
        
        Args:
            model_name: Имя модели ('YOLOv11', 'U-Net', 'nnU-Net')
            model_path: Путь к файлу с весами модели
            
        Returns:
            bool: True если модель успешно загружена, иначе False
        """
        try:
            # Проверка существования файла
            if not os.path.exists(model_path):
                logger.error(f"Файл модели не найден: {model_path}")
                return False
            
            # Проверка доступности модели
            if model_name not in self.available_models:
                logger.error(f"Неизвестная модель: {model_name}")
                return False
            
            # Создание экземпляра модели
            model_class = self.available_models[model_name]
            model_instance = model_class(self.device)
            
            # Загрузка весов
            model_instance.load_model(model_path)
            
            # Сохранение в кэше
            self.models[model_name] = model_instance
            self.model_paths[model_name] = model_path
            
            logger.info(f"Модель {model_name} успешно загружена из {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {model_name}: {str(e)}")
            return False
    
    def predict(self, model_name: str, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Выполнение инференса с использованием указанной модели.
        
        Args:
            model_name: Имя модели
            image: Входное изображение
            
        Returns:
            Маска сегментации или None в случае ошибки
        """
        try:
            # Проверка наличия модели в кэше
            if model_name not in self.models:
                logger.error(f"Модель {model_name} не загружена")
                return None
            
            # Выполнение инференса
            model = self.models[model_name]
            mask = model.predict(image)
            
            logger.info(f"Инференс выполнен с использованием модели {model_name}")
            return mask
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении инференса с моделью {model_name}: {str(e)}")
            return None
    
    def get_available_models(self) -> list:
        """
        Получение списка доступных моделей.
        
        Returns:
            list: Список имен доступных моделей
        """
        return list(self.available_models.keys())
    
    def get_loaded_models(self) -> list:
        """
        Получение списка загруженных моделей.
        
        Returns:
            list: Список имен загруженных моделей
        """
        return list(self.models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """
        Выгрузка модели из памяти.
        
        Args:
            model_name: Имя модели для выгрузки
            
        Returns:
            bool: True если модель успешно выгружена, иначе False
        """
        try:
            if model_name in self.models:
                del self.models[model_name]
                if model_name in self.model_paths:
                    del self.model_paths[model_name]
                
                # Очистка кэша CUDA
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                logger.info(f"Модель {model_name} выгружена из памяти")
                return True
            else:
                logger.warning(f"Модель {model_name} не найдена в кэше")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при выгрузке модели {model_name}: {str(e)}")
            return False
    
    def unload_all_models(self):
        """Выгрузка всех моделей из памяти."""
        model_names = list(self.models.keys())
        for model_name in model_names:
            self.unload_model(model_name)
        
        logger.info("Все модели выгружены из памяти")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о модели.
        
        Args:
            model_name: Имя модели
            
        Returns:
            Словарь с информацией о модели или None в случае ошибки
        """
        try:
            if model_name not in self.models:
                return None
            
            model = self.models[model_name]
            model_path = self.model_paths.get(model_name, "Неизвестно")
            
            return {
                'name': model_name,
                'path': model_path,
                'device': str(self.device),
                'model_type': type(model).__name__
            }
            
        except Exception as e:
            logger.error(f"Ошибка при получении информации о модели {model_name}: {str(e)}")
            return None
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Проверка, загружена ли модель.
        
        Args:
            model_name: Имя модели
            
        Returns:
            bool: True если модель загружена, иначе False
        """
        return model_name in self.models
    
    def scale_mask_for_preview(self, mask: np.ndarray, original_shape: Tuple[int, ...],
                              target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Масштабировать маску под размер превью.
        
        Args:
            mask: Входная маска
            original_shape: Оригинальный размер
            target_shape: Целевой размер (размер превью)
            
        Returns:
            Масштабированная маска
        """
        from InferenceAnd3D.utils.mask_scaler import MaskScaler
        scaler = MaskScaler()
        return scaler.scale_mask(mask, target_shape)
    
    def scale_masks_batch_for_preview(self, masks: Dict[str, np.ndarray],
                                     target_shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        """
        Масштабировать пакет масок под размер превью.
        
        Args:
            masks: Словарь масок
            target_shape: Целевой размер (размер превью)
            
        Returns:
            Словарь масштабированных масок
        """
        from InferenceAnd3D.utils.mask_scaler import MaskScaler
        scaler = MaskScaler()
        return scaler.scale_masks_batch(masks, target_shape)