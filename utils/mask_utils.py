import numpy as np
from scipy import ndimage
from typing import Optional, Tuple, Union
import cv2


def apply_mask_to_volume(volume: np.ndarray, mask: np.ndarray, background_value: float = 0) -> np.ndarray:
    """
    Применяет бинарную маску к 3D объему.
    
    Args:
        volume: 3D массив (D, H, W) - оригинальный объем
        mask: 3D массив (D, H, W) - бинарная маска (0 или 1)
        background_value: Значение для фона (по умолчанию 0)
        
    Returns:
        3D массив с примененной маской
    """
    # Проверяем размеры
    if volume.shape != mask.shape:
        raise ValueError(f"Размеры volume {volume.shape} и mask {mask.shape} не совпадают")
    
    # Применяем маску
    masked_volume = volume.copy()
    if background_value == 0:
        masked_volume = masked_volume * mask
    else:
        masked_volume[mask == 0] = background_value
    
    return masked_volume


def create_masked_slice(slice_img: np.ndarray, mask_slice: np.ndarray, 
                       alpha: float = 0.7, background_value: float = 0) -> np.ndarray:
    """
    Создает полупрозрачное наложение маски на 2D слайс.
    
    Args:
        slice_img: 2D массив - изображение слайса
        mask_slice: 2D массив - маска для слайса
        alpha: Прозрачность маски (0.0 - полностью прозрачная, 1.0 - непрозрачная)
        background_value: Значение для фона
        
    Returns:
        2D массив с наложенной маской
    """
    # Проверяем размеры
    if slice_img.shape != mask_slice.shape:
        raise ValueError(f"Размеры slice_img {slice_img.shape} и mask_slice {mask_slice.shape} не совпадают")
    
    # Если это цветное изображение (3 канала), обрабатываем каждый канал
    if len(slice_img.shape) == 3 and slice_img.shape[2] == 3:
        # Создаем копию изображения
        result = slice_img.copy().astype(np.float32)
        
        # Применяем маску к каждому каналу
        for c in range(3):
            if background_value == 0:
                result[:, :, c] *= mask_slice.astype(np.float32)
            else:
                result[:, :, c] = np.where(mask_slice.astype(bool), result[:, :, c], background_value)
    else:
        # Для одноканального изображения
        result = slice_img.copy().astype(np.float32)
        if background_value == 0:
            result *= mask_slice.astype(np.float32)
        else:
            result = np.where(mask_slice.astype(bool), result, background_value)
    
    return result.astype(slice_img.dtype)


def create_contour_overlay(slice_img: np.ndarray, mask_slice: np.ndarray, 
                          color: Tuple[int, int, int] = (0, 255, 0), 
                          thickness: int = 2) -> np.ndarray:
    """
    Создает контурную обводку маски на 2D слайсе.
    
    Args:
        slice_img: 2D массив - изображение слайса
        mask_slice: 2D массив - маска для слайса
        color: Цвет контура (B, G, R)
        thickness: Толщина линии контура
        
    Returns:
        2D массив с нарисованными контурами
    """
    # Проверяем размеры
    if slice_img.shape != mask_slice.shape:
        raise ValueError(f"Размеры slice_img {slice_img.shape} и mask_slice {slice_img.shape} не совпадают")
    
    # Преобразуем маску в uint8 если нужно
    mask_uint8 = (mask_slice * 255).astype(np.uint8)
    
    # Находим контуры
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Создаем копию изображения
    if len(slice_img.shape) == 2:
        # Для одноканального изображения создаем 3-канальное для отображения контуров
        result = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR) if slice_img.max() <= 255 else slice_img
        if len(result.shape) == 2:
            # Если не удалось преобразовать в BGR, создаем массив вручную
            result = np.stack([slice_img] * 3, axis=-1)
    else:
        result = slice_img.copy()
    
    # Рисуем контуры
    for contour in contours:
        if len(contour) >= 2:  # Убедимся, что есть хотя бы 2 точки для рисования
            cv2.drawContours(result, [contour], -1, color, thickness)
    
    return result


def create_transparent_overlay(slice_img: np.ndarray, mask_slice: np.ndarray, 
                              alpha: float = 0.5, overlay_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    """
    Создает полупрозрачное цветное наложение маски на 2D слайс.
    
    Args:
        slice_img: 2D массив - изображение слайса
        mask_slice: 2D массив - маска для слайса
        alpha: Прозрачность наложения (0.0 - фон, 1.0 - полностью накладывается)
        overlay_color: Цвет наложения (B, G, R)
        
    Returns:
        2D массив с полупрозрачным наложением
    """
    # Проверяем размеры
    if slice_img.shape != mask_slice.shape:
        raise ValueError(f"Размеры slice_img {slice_img.shape} и mask_slice {mask_slice.shape} не совпадают")
    
    # Преобразуем в 3-канальное изображение если необходимо
    if len(slice_img.shape) == 2:
        img_bgr = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR) if slice_img.max() <= 255 else np.stack([slice_img] * 3, axis=-1)
    else:
        img_bgr = slice_img.copy()
    
    # Создаем цветное наложение
    overlay = np.zeros_like(img_bgr)
    for i in range(3):  # RGB каналы
        overlay[:, :, i] = overlay_color[i]
    
    # Применяем маску к наложению
    masked_overlay = np.zeros_like(img_bgr)
    for i in range(3):
        masked_overlay[:, :, i] = overlay[:, :, i] * mask_slice
    
    # Смешиваем оригинальное изображение с наложением
    result = img_bgr.copy()
    result = (1 - alpha) * result + alpha * masked_overlay
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def smooth_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Сглаживает края маски для более плавного наложения.
    
    Args:
        mask: Бинарная маска
        kernel_size: Размер ядра для сглаживания
        
    Returns:
        Сглаженная маска
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    smoothed = cv2.filter2D(mask.astype(np.float32), -1, kernel)
    return smoothed


def extract_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Извлекает самый большой связный компонент из маски.
    
    Args:
        mask: Бинарная маска
        
    Returns:
        Маска с единственным самым большим компонентом
    """
    # Находим все связные компоненты
    labeled, num_features = ndimage.label(mask)
    
    if num_features < 2:
        return mask
    
    # Находим размеры всех компонентов
    component_sizes = ndimage.sum(mask, labeled, range(int(num_features) + 1))
    
    # Находим индекс самого большого компонента (не считая фоновый)
    largest_component_idx = np.argmax(component_sizes[1:]) + 1
    
    # Создаем новую маску только с самым большим компонентом
    largest_component_mask = (labeled == largest_component_idx).astype(np.uint8)
    
    return largest_component_mask