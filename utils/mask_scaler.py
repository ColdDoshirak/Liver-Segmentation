import numpy as np
from scipy.ndimage import zoom
from typing import Tuple, Optional


class MaskScaler:
    """
    Класс для масштабирования масок до заданного разрешения
    """
    
    @staticmethod
    def scale_mask(mask: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Масштабировать маску до заданного размера
        
        Args:
            mask: Входная маска (numpy array)
            target_shape: Целевой размер (height, width) или (depth, height, width)
            
        Returns:
            Масштабированная маска
        """
        if mask.shape == target_shape:
            return mask.copy()
        
        # Рассчитываем факторы масштабирования для каждого измерения
        zoom_factors = []
        for orig_dim, target_dim in zip(mask.shape, target_shape):
            zoom_factors.append(target_dim / float(orig_dim))
        
        # Применяем интерполяцию ближайшего соседа для масок (чтобы сохранить метки классов)
        # Проверяем, чтобы длина zoom_factors совпадала с размерностью маски
        if len(zoom_factors) != mask.ndim:
            # Если размерности не совпадают, корректируем zoom_factors
            if mask.ndim < len(zoom_factors):
                # Убираем лишние факторы
                zoom_factors = zoom_factors[:mask.ndim]
            else:
                # Добавляем факторы для недостающих измерений
                zoom_factors = tuple(list(zoom_factors) + [1.0] * (mask.ndim - len(zoom_factors)))
        
        scaled_mask = zoom(mask, zoom_factors, order=0, mode='nearest')
        
        return scaled_mask.astype(mask.dtype)
    
    @staticmethod
    def scale_masks_batch(masks: dict, target_shape: Tuple[int, ...]) -> dict:
        """
        Масштабировать пакет масок до заданного размера
        
        Args:
            masks: Словарь масок {class_name: mask_array}
            target_shape: Целевой размер
            
        Returns:
            Словарь масштабированных масок
        """
        scaled_masks = {}
        for class_name, mask in masks.items():
            scaled_masks[class_name] = MaskScaler.scale_mask(mask, target_shape)
        return scaled_masks
    
    @staticmethod
    def get_scale_factor(original_shape: Tuple[int, ...], target_shape: Tuple[int, ...]) -> Tuple[float, ...]:
        """
        Получить факторы масштабирования
        
        Args:
            original_shape: Оригинальный размер
            target_shape: Целевой размер
            
        Returns:
            Кортеж факторов масштабирования для каждого измерения
        """
        scale_factors = []
        for orig_dim, target_dim in zip(original_shape, target_shape):
            scale_factors.append(target_dim / float(orig_dim))
        return tuple(scale_factors)
    
    @staticmethod
    def apply_scale_factor_to_coordinates(coords: Tuple[int, ...], scale_factors: Tuple[float, ...]) -> Tuple[int, ...]:
        """
        Применить факторы масштабирования к координатам
        
        Args:
            coords: Координаты (x, y) или (z, y, x)
            scale_factors: Факторы масштабирования
            
        Returns:
            Масштабированные координаты
        """
        scaled_coords = []
        for coord, factor in zip(coords, scale_factors):
            scaled_coords.append(int(coord * factor))
        return tuple(scaled_coords)