import os
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from .preview_cache import PreviewCache
from .mask_scaler import MaskScaler
from typing import TYPE_CHECKING

# Импортируем функции из mask_utils для работы с масками
try:
    from .mask_utils import apply_mask_to_volume, create_masked_slice, create_contour_overlay, create_transparent_overlay, smooth_mask, extract_largest_component
except ImportError:
    from mask_utils import apply_mask_to_volume, create_masked_slice, create_contour_overlay, create_transparent_overlay, smooth_mask, extract_largest_component

if TYPE_CHECKING:
    from InferenceAnd3D.utils.dicom_loader import DicomLoader, MaskedVolumeProcessor


class PreviewManager:
    """
    Центральный менеджер превью изображений
    """
    
    def __init__(self, cache_size: int = 10):
        self.preview_cache = PreviewCache(max_size=cache_size)
        self.mask_scaler = MaskScaler()
        from InferenceAnd3D.utils.dicom_loader import DicomLoader, MaskedVolumeProcessor
        self.dicom_loader = DicomLoader()
        self.masked_volume_processor = MaskedVolumeProcessor()
        self.current_resolution = 'original'
        self.use_preview_for_display = True
        self.available_resolutions = ['original', 128, 256, 512]
        
    def set_resolution(self, resolution: Union[str, int]) -> None:
        """
        Установить разрешение превью
        
        Args:
            resolution: 'original', 128, 256 или 512
        """
        if resolution not in self.available_resolutions:
            raise ValueError(f"Неподдерживаемое разрешение: {resolution}")
        self.current_resolution = resolution
    
    def generate_preview(self, dicom_folder_path: str, resolution: Union[str, int, None] = None, series_id: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Генерировать превью для DICOM серии
        
        Args:
            dicom_folder_path: Путь к папке с DICOM файлами
            series_id: ID DICOM серии (опционально, если None - используется первая серия)
            resolution: Разрешение ('original', 128, 256, 512). Если None, используется текущее
            
        Returns:
            Словарь с превью изображений
        """
        if resolution is None:
            resolution = self.current_resolution
        if resolution is None:  # Дополнительная проверка на случай, если self.current_resolution также None
            resolution = 'original'
        
        # Формируем ключ для кэша, включая series_id, если он предоставлен
        cache_key = f"{dicom_folder_path}_{resolution}_{series_id}" if series_id else f"{dicom_folder_path}_{resolution}"
        
        # Проверяем наличие в кэше
        cached_item = self.preview_cache.get(cache_key)
        if cached_item is not None:
            # Проверяем тип кэшированного элемента
            if isinstance(cached_item, dict):
                return cached_item
            else:
                # Если кэшированный элемент не является словарем, удаляем его и продолжаем
                self.preview_cache.cache.pop(cache_key, None)
        
        # Если series_id None, получаем первый доступный ID
        if series_id is None:
            available_series = self.dicom_loader.get_dicom_series(dicom_folder_path)
            if available_series:
                series_id = available_series[0]
            else:
                raise ValueError(f"Не найдено DICOM серий в папке {dicom_folder_path}")
        
        # Загружаем оригинальные изображения через DicomLoader
        original_sitk_image = self.dicom_loader.load_series(dicom_folder_path, series_id)
        original_image_array = self.dicom_loader.convert_to_numpy(original_sitk_image)
        
        # Создаем превью в зависимости от выбранного разрешения
        if resolution == 'original':
            preview_image_array = original_image_array
        else:
            # Для 3D изображений создаем уменьшенную версию
            preview_image_array = self._downsample_3d_image(original_image_array, resolution)
        
        # Возвращаем как словарь для совместимости, но с 3D массивом
        preview_images = {'preview': preview_image_array}
        
        # Сохраняем в кэш
        self.preview_cache.put(cache_key, preview_images)
        
        return preview_images

    def _downsample_3d_image(self, image_array: np.ndarray, target_resolution: Union[str, int]) -> np.ndarray:
        """
        Даунсэмплировать 3D изображение до заданного разрешения
        
        Args:
            image_array: 3D массив изображения
            target_resolution: Целевое разрешение ('original', 128, 256, 512)
            
        Returns:
            Даунсэмплированный 3D массив
        """
        if target_resolution == 'original' or image_array.ndim != 3:
            return image_array
        
        target_size = int(target_resolution)
        
        # Вычисляем единый фактор масштабирования на основе максимального измерения
        # для сохранения пропорций изображения
        original_shape = image_array.shape
        max_dim = max(original_shape)
        scale_factor = min(target_size / max_dim, 1.0)
        
        # Все измерения масштабируем с одинаковым фактором для сохранения пропорций
        zoom_factors = [scale_factor] * len(original_shape)
        
        try:
            from scipy.ndimage import zoom
            # Используем кубическую интерполяцию для лучшего качества медицинских изображений
            downsampled = zoom(image_array, zoom_factors, order=3)  # cubic interpolation
            return downsampled
        except ImportError:
            # Если scipy недоступен, возвращаем оригинальное изображение
            print("scipy не доступен, возвращаем оригинальное изображение")
            return image_array
    
    def _downsample_images(self, images: Dict[str, np.ndarray], target_size: int) -> Dict[str, np.ndarray]:
        """
        Даунсэмплировать изображения до заданного размера
        
        Args:
            images: Словарь с изображениями
            target_size: Целевой размер (сторона квадрата)
            
        Returns:
            Словарь с даунсэмплированными изображениями
        """
        downsampled_images = {}
        for slice_name, image in images.items():
            if len(image.shape) == 2:  # 2D изображение
                h, w = image.shape
                if max(h, w) > target_size:
                    # Масштабируем так, чтобы максимальная сторона была target_size
                    scale_factor = target_size / max(h, w)
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    from InferenceAnd3D.utils.dicom_loader import downsample_image
                    downsampled_image = downsample_image(image, (new_h, new_w))
                else:
                    downsampled_image = image
            else:
                downsampled_image = image  # Необработанный случай для 3D данных
                
            downsampled_images[slice_name] = downsampled_image
        
        return downsampled_images
    
    def scale_masks_for_preview(self, masks: Dict[str, np.ndarray], original_shape: Tuple[int, ...],
                               target_shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        """
        Масштабировать маски под размер превью
        
        Args:
            masks: Словарь масок
            original_shape: Оригинальный размер
            target_shape: Целевой размер (размер превью)
            
        Returns:
            Словарь масштабированных масок
        """
        return self.mask_scaler.scale_masks_batch(masks, target_shape)
    
    def get_scaled_mask(self, mask: np.ndarray, original_shape: Tuple[int, ...],
                       target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Масштабировать одну маску под размер превью
        
        Args:
            mask: Входная маска
            original_shape: Оригинальный размер
            target_shape: Целевой размер (размер превью)
            
        Returns:
            Масштабированная маска
        """
        return self.mask_scaler.scale_mask(mask, target_shape)
    
    def get_preview_shape_for_resolution(self, original_shape: Tuple[int, ...], resolution: Union[str, int]) -> Tuple[int, ...]:
        """
        Получить размер превью для заданного разрешения
        
        Args:
            original_shape: Оригинальный размер изображения
            resolution: Целевое разрешение ('original', 128, 256, 512)
            
        Returns:
            Размер превью
        """
        if resolution == 'original':
            return original_shape
        
        target_size = int(resolution)
        
        # Вычисляем единый фактор масштабирования на основе максимального измерения
        max_dim = max(original_shape)
        scale_factor = min(target_size / max_dim, 1.0)
        
        # Вычисляем новые размеры с сохранением пропорций
        new_shape = tuple(int(dim * scale_factor) for dim in original_shape)
        
        return new_shape
    
    def update_use_preview_setting(self, use_preview: bool) -> None:
        """
        Обновить настройку использования превью для отображения
        
        Args:
            use_preview: True если использовать превью для отображения
        """
        self.use_preview_for_display = use_preview
    
    def clear_cache(self) -> None:
        """Очистить кэш превью"""
        self.preview_cache.clear()
    
    def get_cached_keys(self) -> List[str]:
        """Получить список ключей в кэше"""
        return self.preview_cache.keys()
    
    def get_cache_size(self) -> int:
        """Получить текущий размер кэша"""
        return self.preview_cache.size()
    
    def is_preview_available(self, dicom_series_path: str, resolution: Union[str, int, None] = None, series_id: Optional[str] = None) -> bool:
        """
        Проверить доступность превью в кэше
        
        Args:
            dicom_series_path: Путь к DICOM серии
            resolution: Разрешение ('original', 128, 256, 512). Если None, используется текущее
            series_id: ID DICOM серии (опционально, если None - используется только путь и разрешение)
            
        Returns:
            True если превью доступно в кэше
        """
        if resolution is None:
            resolution = self.current_resolution
        if resolution is None:  # Дополнительная проверка на случай, если self.current_resolution также None
            resolution = 'original'
        
        cache_key = f"{dicom_series_path}_{resolution}_{series_id}" if series_id else f"{dicom_series_path}_{resolution}"
        return self.preview_cache.contains(cache_key)
    
    def generate_masked_preview(self, volume: np.ndarray, mask: np.ndarray, 
                              mode: str = 'multiply', alpha: float = 0.7, 
                              overlay_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """
        Генерирует маскированное превью 3D объема.
        
        Args:
            volume: 3D массив - оригинальный объем
            mask: 3D массив - бинарная маска
            mode: Режим маскирования ('multiply', 'overlay', 'contour', 'transparent')
            alpha: Прозрачность для режимов overlay и transparent
            overlay_color: Цвет для прозрачного наложения
            
        Returns:
            3D массив - маскированный объем
        """
        return self.masked_volume_processor.create_masked_volume(volume, mask, mode, background_value=0)
    
    def generate_masked_slice_preview(self, volume: np.ndarray, mask: np.ndarray, slice_idx: int,
                                    mode: str = 'multiply', alpha: float = 0.7, 
                                    overlay_color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """
        Генерирует маскированный слайс из 3D объема для превью.
        
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
        from InferenceAnd3D.utils.dicom_loader import create_masked_slice_from_volume
        return create_masked_slice_from_volume(volume, mask, slice_idx, mode, alpha, overlay_color)