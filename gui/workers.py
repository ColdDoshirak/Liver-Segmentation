"""
Модуль с классами для многопоточной обработки в GUI приложении.

Этот модуль содержит классы-наследники QThread для выполнения длительных операций
в отдельных потоках, чтобы избежать зависания интерфейса пользователя.
"""

import os
import time
import numpy as np
import torch
import SimpleITK as sitk
from typing import Dict, Optional, Union, Tuple, Any
import logging

# Импорты PyQt6 с обработкой ошибок
# Ошибки Pylance здесь связаны с отсутствием PyQt в среде разработки, но код будет работать при установке
try:
    from PyQt6.QtCore import QThread, pyqtSignal, QObject
except ImportError:
    try:
        from PyQt5.QtCore import QThread, pyqtSignal, QObject
    except ImportError:
        raise ImportError("Требуется PyQt6 или PyQt5. Установите с помощью: pip install PyQt6")

# Импорты из локальных модулей
try:
    # Абсолютные импорты от корня проекта
    from InferenceAnd3D.utils.dicom_loader import DicomLoader, preprocess_for_model, postprocess_mask
    from InferenceAnd3D.models.model_manager import ModelManager
    from InferenceAnd3D.config import get_model_path, get_device
except ImportError:
    # Альтернативные абсолютные импорты (если InferenceAnd3D уже в sys.path)
    from utils.dicom_loader import DicomLoader, preprocess_for_model, postprocess_mask
    from models.model_manager import ModelManager
    from config import get_model_path, get_device

# Настройка логирования с поддержкой кириллицы и Unicode
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DicomLoaderWorker(QThread):
    """
    Класс для загрузки DICOM серий в отдельном потоке.
    
    Этот класс выполняет загрузку DICOM файлов в фоновом потоке,
    позволяя GUI оставаться отзывчивым во время операции.
    """
    
    # Сигналы
    progress_updated = pyqtSignal(int)  # Обновление прогресса (0-100)
    series_loaded = pyqtSignal(dict)    # Загруженная серия (словарь с данными)
    error_occurred = pyqtSignal(str)    # Сообщение об ошибке
    
    def __init__(self, parent=None):
        """
        Инициализация рабочего потока для загрузки DICOM.
        
        Args:
            parent: Родительский QObject
        """
        super(DicomLoaderWorker, self).__init__(parent)
        self.folder_path = ""
        self.series_id = ""
        self.dicom_loader = DicomLoader()
        self._is_running = False
        
    def load_series(self, folder_path: str, series_id: str):
        """
        Установка параметров для загрузки серии.
        
        Args:
            folder_path (str): Путь к папке с DICOM файлами
            series_id (str): ID серии для загрузки
        """
        self.folder_path = folder_path
        self.series_id = series_id
        
    def run(self):
        """
        Основной метод потока - выполняет загрузку DICOM серии.
        """
        self._is_running = True
        
        try:
            # Эмитация начала загрузки
            self.progress_updated.emit(0)
            
            # Проверка существования папки
            if not os.path.exists(self.folder_path):
                raise FileNotFoundError(f"Папка не существует: {self.folder_path}")
                
            # Проверка, что путь является директорией
            if not os.path.isdir(self.folder_path):
                raise ValueError(f"Указанный путь не является директорией: {self.folder_path}")
            
            # Эмитация прогресса
            self.progress_updated.emit(10)
            
            # Получение информации о серии
            series_info = self.dicom_loader.get_series_info(self.folder_path, self.series_id)
            
            # Эмитация прогресса
            self.progress_updated.emit(30)
            
            # Загрузка серии
            sitk_image = self.dicom_loader.load_series(self.folder_path, self.series_id)
            
            # Эмитация прогресса
            self.progress_updated.emit(60)
            
            # Конвертация в numpy массив
            image_array = self.dicom_loader.convert_to_numpy(sitk_image)
            
            # Эмитация прогресса
            self.progress_updated.emit(80)
            
            # Получение объема вокселя
            voxel_volume = self.dicom_loader.get_voxel_volume(sitk_image)
            
            # Формирование результата
            result = {
                'sitk_image': sitk_image,
                'image_array': image_array,
                'series_info': series_info,
                'voxel_volume': voxel_volume,
                'folder_path': self.folder_path,
                'series_id': self.series_id
            }
            
            # Проверка, что поток не был остановлен
            if self._is_running:
                self.progress_updated.emit(10)
                self.series_loaded.emit(result)
                
        except Exception as e:
            if self._is_running:
                logger.error(f"Ошибка при загрузке DICOM серии: {str(e)}")
                self.error_occurred.emit(f"Ошибка при загрузке DICOM серии: {str(e)}")
                
    def stop(self):
        """
        Остановка выполнения потока.
        """
        self._is_running = False


class SegmentationWorker(QThread):
    """
    Класс для запуска инференса моделей сегментации в отдельном потоке.
    
    Этот класс выполняет сегментацию изображений с использованием различных моделей
    в фоновом потоке, позволяя GUI оставаться отзывчивым во время операции.
    """
    
    # Сигналы
    progress_updated = pyqtSignal(int)           # Обновление прогресса (0-100)
    segmentation_completed = pyqtSignal(object)  # Результат сегментации (маска или словарь с маской и типом модели)
    error_occurred = pyqtSignal(str)            # Сообщение об ошибке
    volume_calculated = pyqtSignal(float)       # Рассчитанный объем
    
    def __init__(self, parent=None):
        """
        Инициализация рабочего потока для сегментации.
        
        Args:
            parent: Родительский QObject
        """
        super(SegmentationWorker, self).__init__(parent)
        self.image_array = None
        self.model_type = ""
        self.model_path = ""
        self.device = torch.device(get_device())
        self._is_running = False
        
        # Инициализация менеджера моделей
        self.model_manager = ModelManager(self.device)
        
    def segment_image(self, image_array: np.ndarray, model_type: str, model_path: Optional[str] = None):
        """
        Установка параметров для сегментации изображения.
        
        Args:
            image_array (np.ndarray): Входное изображение в формате numpy
            model_type (str): Тип модели ('YOLOv11', 'U-Net', 'nnU-Net')
            model_path (str): Путь к файлу с весами модели (опционально)
        """
        self.image_array = image_array
        self.model_type = model_type
        
        # Если путь не указан, используем путь из конфигурации
        if model_path is None:
            self.model_path = get_model_path(model_type)
        else:
            self.model_path = model_path
        
    def run(self):
        """
        Основной метод потока - выполняет сегментацию изображения.
        """
        self._is_running = True
        
        try:
            # Эмитация начала сегментации
            self.progress_updated.emit(0)
            
            # Проверка входных данных
            if self.image_array is None:
                raise ValueError("Изображение не загружено")
                
            if self.model_path is None or not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
            
            # Эмитация прогресса
            self.progress_updated.emit(10)
            
            # Предобработка изображения в зависимости от типа модели
            processed_image = preprocess_for_model(self.image_array, self.model_type)
            
            # Эмитация прогресса
            self.progress_updated.emit(20)
            
            # Проверка, загружена ли модель
            if not self.model_manager.is_model_loaded(self.model_type):
                # Загрузка модели через менеджер
                success = self.model_manager.load_model(self.model_type, self.model_path or "")
                if not success:
                    raise ValueError(f"Не удалось загрузить модель {self.model_type}")
            
            # Эмитация прогресса
            self.progress_updated.emit(40)
            
            # Выполнение инференса через менеджер моделей
            segmentation_mask = self.model_manager.predict(self.model_type, processed_image)
            
            if segmentation_mask is None:
                raise ValueError(f"Не удалось выполнить инференс с моделью {self.model_type}")
            
            # Эмитация прогресса
            self.progress_updated.emit(80)
            
            # Постобработка маски
            if self._is_running:
                segmentation_mask = postprocess_mask(segmentation_mask)
                
                # Эмитация прогресса
                self.progress_updated.emit(90)
                
                # Расчет объема (если 3D)
                if len(segmentation_mask.shape) == 3:
                    # Подсчет вокселей печени
                    liver_voxels = np.sum(segmentation_mask > 0)
                    
                    # Расчет объема в см³ (предполагаем, что воксели имеют размер 1x1x1 мм)
                    volume_cm3 = liver_voxels / 1000.0
                    
                    # Эмитация рассчитанного объема
                    self.volume_calculated.emit(volume_cm3)
                
                # Эмитация прогресса
                self.progress_updated.emit(10)
                
                # Эмитация результата - теперь передаем словарь с маской и типом модели
                result = {
                    'mask': segmentation_mask,
                    'model_type': self.model_type
                }
                self.segmentation_completed.emit(result)
                
        except Exception as e:
            if self._is_running:
                logger.error(f"Ошибка при сегментации изображения: {str(e)}")
                self.error_occurred.emit(f"Ошибка при сегментации изображения: {str(e)}")
                
    def stop(self):
        """
        Остановка выполнения потока.
        """
        self._is_running = False


class VolumeCalculationWorker(QThread):
    """
    Класс для расчета объема печени на основе маски сегментации.
    
    Этот класс выполняет расчет объема в фоновом потоке,
    позволяя GUI оставаться отзывчивым во время операции.
    """
    
    # Сигналы
    volume_calculated = pyqtSignal(float)  # Рассчитанный объем
    error_occurred = pyqtSignal(str)       # Сообщение об ошибке
    
    def __init__(self, parent=None):
        """
        Инициализация рабочего потока для расчета объема.
        
        Args:
            parent: Родительский QObject
        """
        super(VolumeCalculationWorker, self).__init__(parent)
        self.mask_array = None
        self.voxel_volume = 1.0  # мм³ по умолчанию
        self._is_running = False
        
    def calculate_volume(self, mask_array: np.ndarray, voxel_volume: float):
        """
        Установка параметров для расчета объема.
        
        Args:
            mask_array (np.ndarray): Маска сегментации
            voxel_volume (float): Объем одного вокселя в мм³
        """
        self.mask_array = mask_array
        self.voxel_volume = voxel_volume
        
    def run(self):
        """
        Основной метод потока - выполняет расчет объема.
        """
        self._is_running = True
        
        try:
            # Проверка входных данных
            if self.mask_array is None:
                raise ValueError("Маска сегментации не загружена")
                
            # Подсчет вокселей печени
            liver_voxels = np.sum(self.mask_array > 0)
            
            # Расчет объема в мм³
            volume_mm3 = liver_voxels * self.voxel_volume
            
            # Конвертация в см³
            volume_cm3 = volume_mm3 / 1000.0
            
            # Проверка, что поток не был остановлен
            if self._is_running:
                self.volume_calculated.emit(volume_cm3)
                
        except Exception as e:
            if self._is_running:
                logger.error(f"Ошибка при расчете объема: {str(e)}")
                self.error_occurred.emit(f"Ошибка при расчете объема: {str(e)}")
                
    def stop(self):
        """
        Остановка выполнения потока.
        """
        self._is_running = False


class DisplayUpdateWorker(QThread):
    """
    Класс для асинхронного обновления отображения 3D визуализации.
    
    Этот класс выполняет обновление отображения в отдельном потоке,
    позволяя GUI оставаться отзывчивым во время операций с большими изображениями.
    """
    
    # Сигналы
    display_updated = pyqtSignal(object)  # Обновленные данные для отображения
    error_occurred = pyqtSignal(str)      # Сообщение об ошибке
    
    def __init__(self, parent=None):
        """
        Инициализация рабочего потока для обновления отображения.
        
        Args:
            parent: Родительский QObject
        """
        super(DisplayUpdateWorker, self).__init__(parent)
        self.image_data = None
        self.mask_data = None
        self.window_center = 0
        self.window_width = 400
        self.show_ct_image = True
        self.show_mask = True
        self.mask_opacity = 0.5
        self.show_masked_region = False
        self.mask_display_mode = 'multiply'
        self._is_running = False
        
    def set_parameters(self, image_data: np.ndarray, mask_data: Optional[np.ndarray] = None,
                      window_center: int = 0, window_width: int = 400,
                      show_ct_image: bool = True, show_mask: bool = True,
                      mask_opacity: float = 0.5, show_masked_region: bool = False,
                      mask_display_mode: str = 'multiply'):
        """
        Установка параметров для обновления отображения.
        
        Args:
            image_data: Данные КТ изображения
            mask_data: Данные маски сегментации
            window_center: Центр окна контраста
            window_width: Ширина окна контраста
            show_ct_image: Флаг отображения КТ изображения
            show_mask: Флаг отображения маски
            mask_opacity: Прозрачность маски
            show_masked_region: Флаг отображения выделенной области
            mask_display_mode: Режим отображения маски ('multiply', 'overlay', 'contour')
        """
        self.image_data = image_data
        self.mask_data = mask_data
        self.window_center = window_center
        self.window_width = window_width
        self.show_ct_image = show_ct_image
        self.show_mask = show_mask
        self.mask_opacity = mask_opacity
        self.show_masked_region = show_masked_region
        self.mask_display_mode = mask_display_mode
        
    def apply_window_level(self, image_data: np.ndarray) -> np.ndarray:
        """
        Оптимизированное применение настроек окна/уровня к изображению.
        
        Args:
            image_data: Исходные данные изображения
            
        Returns:
            np.ndarray: Изображение с примененными настройками контраста
        """
        # Векторизованное вычисление границ окна
        window_min = self.window_center - self.window_width / 2
        window_max = self.window_center + self.window_width / 2
        
        # Векторизованное применение окна/уровня
        windowed_image = np.clip(image_data, window_min, window_max)
        
        # Векторизованная нормализация в диапазон [0, 1]
        if self.window_width > 0:
            # Используем in-place операции для экономии памяти
            windowed_image -= window_min
            windowed_image /= self.window_width
        else:
            windowed_image = np.zeros_like(windowed_image)
            
        return windowed_image
        
    def run(self):
        """
        Оптимизированный основной метод потока - выполняет обновление отображения.
        """
        self._is_running = True
        start_time = time.time()
        
        try:
            # Проверка входных данных
            if self.image_data is None:
                raise ValueError("Данные изображения не загружены")
                
            # Подготовка результата
            result = {
                'image_data': None,
                'mask_data': self.mask_data,
                'show_ct_image': self.show_ct_image,
                'show_mask': self.show_mask,
                'mask_opacity': self.mask_opacity,
                'show_masked_region': self.show_masked_region,
                'mask_display_mode': self.mask_display_mode
            }
            
            # Оптимизированная обработка КТ изображения, если включено
            if self.show_ct_image:
                # Проверка размера изображения для определения стратегии обработки
                image_size_mb = self.image_data.nbytes / (1024 * 1024)
                
                if image_size_mb > 50:  # Если изображение больше 50 МБ, используем прогрессивную обработку
                    result['image_data'] = self._process_large_volume(self.image_data)
                else:
                    # Примение настроек окна/уровня
                    windowed_volume = self.apply_window_level(self.image_data)
                    result['image_data'] = windowed_volume
                
            # Проверка, что поток не был остановлен
            if self._is_running:
                processing_time = time.time() - start_time
                logger.info(f"Обновление отображения выполнено за {processing_time:.3f}с")
                self.display_updated.emit(result)
                
        except Exception as e:
            if self._is_running:
                logger.error(f"Ошибка при обновлении отображения: {str(e)}")
                self.error_occurred.emit(f"Ошибка при обновлении отображения: {str(e)}")
    
    def _process_large_volume(self, volume_data: np.ndarray) -> np.ndarray:
        """
        Оптимизированная обработка больших объемов данных.
        
        Args:
            volume_data: 3D массив данных объема
            
        Returns:
            Обработанный 3D массив данных
        """
        # Для больших объемов применяем обработку без уменьшения разрешения
        # Удаляем автоматическое уменьшение разрешения, чтобы использовать выбранное пользователем разрешение
        
        # Стандартная обработка полного объема
        return self.apply_window_level(volume_data)
                
    def stop(self):
        """
        Остановка выполнения потока.
        """
        self._is_running = False


class PreviewGenerationWorker(QThread):
    """
    Класс для генерации превью изображений в отдельном потоке.
    
    Этот класс выполняет генерацию превью в фоновом потоке,
    позволяя GUI оставаться отзывчивым во время операции.
    """
    
    # Сигналы
    progress_updated = pyqtSignal(int)           # Обновление прогресса (0-100)
    preview_generated = pyqtSignal(dict)         # Сгенерированные превью
    error_occurred = pyqtSignal(str)             # Сообщение об ошибке
    
    def __init__(self, parent=None):
        """
        Инициализация рабочего потока для генерации превью.
        
        Args:
            parent: Родительский QObject
        """
        super(PreviewGenerationWorker, self).__init__(parent)
        self.dicom_folder = ""
        self.series_id = ""
        self.resolution = "original"
        self._is_running = False
        
    def generate_preview(self, dicom_folder: str, series_id: str, resolution: Union[str, int]):
        """
        Установка параметров для генерации превью.
        
        Args:
            dicom_folder: Путь к папке с DICOM файлами
            series_id: ID серии для генерации превью
            resolution: Разрешение превью ('original', 128, 256, 512)
        """
        self.dicom_folder = dicom_folder
        self.series_id = series_id
        self.resolution = resolution
        
    def run(self):
        """
        Основной метод потока - выполняет генерацию превью.
        """
        self._is_running = True
        
        try:
            # Эмитация начала генерации
            self.progress_updated.emit(0)
            
            # Проверка входных данных
            if not self.dicom_folder or not self.series_id:
                raise ValueError("Не указаны путь к DICOM или ID серии")
                
            # Эмитация прогресса
            self.progress_updated.emit(10)
            
            # Импортируем PreviewManager
            from InferenceAnd3D.utils.preview_manager import PreviewManager
            
            # Создаем экземпляр PreviewManager
            preview_manager = PreviewManager()
            
            # Эмитация прогресса
            self.progress_updated.emit(20)
            
            # Устанавливаем разрешение
            preview_manager.set_resolution(self.resolution)
            
            # Эмитация прогресса
            self.progress_updated.emit(30)
            
            # Генерируем превью
            # Вместо объединения dicom_folder и series_id, передаем их отдельно
            # generate_preview ожидает dicom_folder_path и необязательный series_id
            preview_data = preview_manager.generate_preview(self.dicom_folder, resolution=self.resolution, series_id=self.series_id)
            
            # Эмитация прогресса
            self.progress_updated.emit(80)
            
            # Проверка, что поток не был остановлен
            if self._is_running:
                self.progress_updated.emit(100)
                self.preview_generated.emit(preview_data)
                
        except Exception as e:
            if self._is_running:
                logger.error(f"Ошибка при генерации превью: {str(e)}")
                self.error_occurred.emit(f"Ошибка при генерации превью: {str(e)}")
                
    def stop(self):
        """
        Остановка выполнения потока.
        """
        self._is_running = False


class MaskApplicationWorker(QThread):
    """
    Класс для асинхронного применения маски к объему в отдельном потоке.
    
    Этот класс выполняет операции маскирования в фоновом потоке,
    позволяя GUI оставаться отзывчивым во время операций с большими изображениями.
    """
    
    # Сигналы
    mask_applied = pyqtSignal(object)      # Результат применения маски (маскированный объем)
    progress_updated = pyqtSignal(int)     # Обновление прогресса (0-10)
    error_occurred = pyqtSignal(str)       # Сообщение об ошибке
    
    def __init__(self, parent=None):
        """
        Инициализация рабочего потока для применения маски.
        
        Args:
            parent: Родительский QObject
        """
        super(MaskApplicationWorker, self).__init__(parent)
        self.image_data = None
        self.mask_data = None
        self.mode = 'multiply'
        self.background_value = 0
        self._is_running = False
        
    def apply_mask(self, image_data: np.ndarray, mask_data: np.ndarray, mode: str = 'multiply', background_value: float = 0):
        """
        Установка параметров для применения маски.
        
        Args:
            image_data: Данные изображения для маскирования
            mask_data: Данные маски
            mode: Режим маскирования ('multiply', 'overlay', 'contour', 'transparent')
            background_value: Значение для фона
        """
        self.image_data = image_data
        self.mask_data = mask_data
        self.mode = mode
        self.background_value = background_value
        
    def run(self):
        """
        Основной метод потока - выполняет применение маски к объему.
        """
        self._is_running = True
        
        try:
            # Проверка входных данных
            if self.image_data is None or self.mask_data is None:
                raise ValueError("Данные изображения или маски не загружены")
                
            # Проверка совпадения размеров
            if self.image_data.shape != self.mask_data.shape:
                raise ValueError(f"Размеры изображения {self.image_data.shape} и маски {self.mask_data.shape} не совпадают")
            
            # Эмитация начала процесса
            self.progress_updated.emit(0)
            
            # Импортируем функции из mask_utils
            from InferenceAnd3D.utils.mask_utils import apply_mask_to_volume, create_masked_slice, create_contour_overlay, create_transparent_overlay
            
            # Применяем маску в зависимости от режима
            if self.mode == 'multiply':
                result = apply_mask_to_volume(self.image_data, self.mask_data, self.background_value)
            elif self.mode == 'overlay':
                # Для режима наложения создаем полупрозрачное изображение
                result = self.image_data.copy()
                for i in range(result.shape[0]):
                    result[i] = create_masked_slice(result[i], self.mask_data[i], alpha=0.7, background_value=self.background_value)
            elif self.mode == 'contour':
                # Для контурного режима создаем изображение с контурами
                result = self.image_data.copy()
                for i in range(result.shape[0]):
                    result[i] = create_contour_overlay(result[i], self.mask_data[i])
            elif self.mode == 'transparent':
                # Для прозрачного наложения
                result = self.image_data.copy()
                for i in range(result.shape[0]):
                    result[i] = create_transparent_overlay(result[i], self.mask_data[i], alpha=0.5)
            else:
                raise ValueError(f"Неподдерживаемый режим маскирования: {self.mode}")
            
            # Эмитация прогресса
            self.progress_updated.emit(100)
            
            # Проверка, что поток не был остановлен
            if self._is_running:
                self.mask_applied.emit(result)
                
        except Exception as e:
            if self._is_running:
                logger.error(f"Ошибка при применении маски: {str(e)}")
                self.error_occurred.emit(f"Ошибка при применении маски: {str(e)}")
                
    def stop(self):
        """
        Остановка выполнения потока.
        """
        self._is_running = False