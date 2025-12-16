"""
Модуль для прогрессивной загрузки и обработки больших изображений.

Этот модуль предоставляет классы и функции для асинхронной загрузки,
обработки по частям и приоритетной обработки видимых областей.
"""

import numpy as np
import time
import logging
from typing import Optional, Callable, Tuple, List, Dict, Any
from threading import Thread, Event
from queue import Queue, PriorityQueue
from dataclasses import dataclass, field
from enum import Enum

# Настройка логирования
logger = logging.getLogger(__name__)


class ProcessingPriority(Enum):
    """Приоритеты обработки изображений."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class ProcessingTask:
    """Задача для обработки изображения."""
    priority: ProcessingPriority
    task_id: str
    image_data: np.ndarray
    callback: Callable
    region: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressiveImageLoader:
    """
    Класс для прогрессивной загрузки и обработки больших изображений.
    
    Позволяет загружать и обрабатывать изображения по частям,
    приоритизировать видимые области и выполнять асинхронную обработку.
    """
    
    def __init__(self, max_workers: int = 2, chunk_size: int = 256):
        """
        Инициализация прогрессивного загрузчика.
        
        Args:
            max_workers: Максимальное количество рабочих потоков
            chunk_size: Размер блока для обработки изображений
        """
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.task_queue = PriorityQueue()
        self.workers: List[Thread] = []
        self.stop_event = Event()
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        
        # Запуск рабочих потоков
        self._start_workers()
        
        logger.info(f"Прогрессивный загрузчик инициализирован с {max_workers} рабочими потоками")
    
    def _start_workers(self):
        """Запуск рабочих потоков."""
        for i in range(self.max_workers):
            worker = Thread(target=self._worker_loop, name=f"ProgressiveLoader-{i}")
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Основной цикл рабочего потока."""
        while not self.stop_event.is_set():
            try:
                # Получение задачи из очереди с таймаутом
                priority, task_id, task = self.task_queue.get(timeout=0.1)
                
                # Добавление задачи в активные
                self.active_tasks[task_id] = task
                
                # Обработка задачи
                try:
                    result = self._process_task(task)
                    
                    # Сохранение результата
                    self.completed_tasks[task_id] = result
                    
                    # Вызов callback функции
                    if task.callback:
                        task.callback(result)
                    
                    logger.info(f"Задача {task_id} выполнена успешно")
                except Exception as e:
                    logger.error(f"Ошибка при выполнении задачи {task_id}: {str(e)}")
                finally:
                    # Удаление из активных
                    self.active_tasks.pop(task_id, None)
                    self.task_queue.task_done()
                    
            except:
                # Прерывание при таймауте - нормальная ситуация
                continue
    
    def _process_task(self, task: ProcessingTask) -> Any:
        """
        Обработка отдельной задачи.
        
        Args:
            task: Задача для обработки
            
        Returns:
            Результат обработки
        """
        start_time = time.time()
        
        # Если указана область, обрабатываем только её
        if task.region:
            x, y, width, height = task.region
            image_chunk = task.image_data[y:y+height, x:x+width]
        else:
            image_chunk = task.image_data
        
        # Обработка в зависимости от типа задачи
        if task.metadata.get('operation') == 'window_level':
            result = self._apply_window_level(
                image_chunk, 
                task.metadata.get('window_center', 0),
                task.metadata.get('window_width', 400)
            )
        elif task.metadata.get('operation') == 'downsample':
            scale_factor = task.metadata.get('scale_factor', 0.5)
            result = self._downsample_image(image_chunk, scale_factor)
        elif task.metadata.get('operation') == 'segmentation':
            model_type = task.metadata.get('model_type', 'U-Net')
            result = self._segment_image(image_chunk, model_type)
        else:
            # По умолчанию возвращаем исходные данные
            result = image_chunk
        
        processing_time = time.time() - start_time
        logger.info(f"Задача {task.task_id} обработана за {processing_time:.3f}с")
        
        return result
    
    def _apply_window_level(self, image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
        """
        Применение настроек окна/уровня к изображению.
        
        Args:
            image: Входное изображение
            window_center: Центр окна
            window_width: Ширина окна
            
        Returns:
            Изображение с примененными настройками
        """
        # Векторизованное применение окна/уровня
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        windowed_image = np.clip(image, window_min, window_max)
        
        if window_width > 0:
            windowed_image = (windowed_image - window_min) / window_width
        else:
            windowed_image = np.zeros_like(windowed_image)
            
        return windowed_image
    
    def _downsample_image(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Уменьшение разрешения изображения.
        
        Args:
            image: Входное изображение
            scale_factor: Коэффициент уменьшения
            
        Returns:
            Уменьшенное изображение
        """
        try:
            from scipy import ndimage
            return ndimage.zoom(image, scale_factor, order=1)
        except ImportError:
            # Если scipy недоступен, используем простое уменьшение
            new_shape = tuple(int(dim * scale_factor) for dim in image.shape)
            if len(image.shape) == 2:
                # Для 2D изображений
                y_indices = np.linspace(0, image.shape[0]-1, new_shape[0]).astype(int)
                x_indices = np.linspace(0, image.shape[1]-1, new_shape[1]).astype(int)
                return image[np.ix_(y_indices, x_indices)]
            else:
                # Для 3D изображений
                z_indices = np.linspace(0, image.shape[0]-1, new_shape[0]).astype(int)
                y_indices = np.linspace(0, image.shape[1]-1, new_shape[1]).astype(int)
                x_indices = np.linspace(0, image.shape[2]-1, new_shape[2]).astype(int)
                return image[np.ix_(z_indices, y_indices, x_indices)]
    
    def _segment_image(self, image: np.ndarray, model_type: str) -> np.ndarray:
        """
        Сегментация изображения.
        
        Args:
            image: Входное изображение
            model_type: Тип модели сегментации
            
        Returns:
            Маска сегментации
        """
        # Импорт здесь для избежания циклических зависимостей
        try:
            from ..models.model_manager import ModelManager
            from ..config import get_model_path
            
            # Создание менеджера моделей
            model_manager = ModelManager()
            
            # Загрузка модели
            model_path = get_model_path(model_type)
            if model_path and model_manager.load_model(model_type, model_path):
                # Выполнение сегментации
                mask = model_manager.predict(model_type, image)
                return mask if mask is not None else np.zeros_like(image, dtype=np.uint8)
            else:
                logger.error(f"Не удалось загрузить модель {model_type}")
                return np.zeros_like(image, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Ошибка при сегментации: {str(e)}")
            return np.zeros_like(image, dtype=np.uint8)
    
    def add_task(self, task: ProcessingTask) -> str:
        """
        Добавление задачи в очередь обработки.
        
        Args:
            task: Задача для обработки
            
        Returns:
            ID задачи
        """
        # Добавление в приоритетную очередь
        self.task_queue.put((task.priority.value, task.task_id, task))
        logger.info(f"Задача {task.task_id} добавлена в очередь с приоритетом {task.priority.name}")
        return task.task_id
    
    def add_window_level_task(self, image_data: np.ndarray, window_center: int, window_width: int,
                           callback: Optional[Callable] = None, priority: ProcessingPriority = ProcessingPriority.NORMAL,
                           region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Добавление задачи применения настроек окна/уровня.
        
        Args:
            image_data: Данные изображения
            window_center: Центр окна
            window_width: Ширина окна
            callback: Функция обратного вызова
            priority: Приоритет задачи
            region: Область обработки (x, y, width, height)
            
        Returns:
            ID задачи
        """
        task_id = f"window_level_{int(time.time() * 1000)}"
        
        task = ProcessingTask(
            priority=priority,
            task_id=task_id,
            image_data=image_data,
            callback=callback or (lambda x: None),
            region=region,
            metadata={
                'operation': 'window_level',
                'window_center': window_center,
                'window_width': window_width
            }
        )
        
        return self.add_task(task)
    
    def add_downsample_task(self, image_data: np.ndarray, scale_factor: float,
                          callback: Optional[Callable] = None, priority: ProcessingPriority = ProcessingPriority.NORMAL,
                          region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Добавление задачи уменьшения разрешения.
        
        Args:
            image_data: Данные изображения
            scale_factor: Коэффициент уменьшения
            callback: Функция обратного вызова
            priority: Приоритет задачи
            region: Область обработки (x, y, width, height)
            
        Returns:
            ID задачи
        """
        task_id = f"downsample_{int(time.time() * 1000)}"
        
        task = ProcessingTask(
            priority=priority,
            task_id=task_id,
            image_data=image_data,
            callback=callback or (lambda x: None),
            region=region,
            metadata={
                'operation': 'downsample',
                'scale_factor': scale_factor
            }
        )
        
        return self.add_task(task)
    
    def add_segmentation_task(self, image_data: np.ndarray, model_type: str,
                           callback: Optional[Callable] = None, priority: ProcessingPriority = ProcessingPriority.NORMAL,
                           region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Добавление задачи сегментации.
        
        Args:
            image_data: Данные изображения
            model_type: Тип модели сегментации
            callback: Функция обратного вызова
            priority: Приоритет задачи
            region: Область обработки (x, y, width, height)
            
        Returns:
            ID задачи
        """
        task_id = f"segmentation_{int(time.time() * 1000)}"
        
        task = ProcessingTask(
            priority=priority,
            task_id=task_id,
            image_data=image_data,
            callback=callback or (lambda x: None),
            region=region,
            metadata={
                'operation': 'segmentation',
                'model_type': model_type
            }
        )
        
        return self.add_task(task)
    
    def get_task_status(self, task_id: str) -> Tuple[str, Optional[Any]]:
        """
        Получение статуса задачи.
        
        Args:
            task_id: ID задачи
            
        Returns:
            Кортеж (статус, результат)
        """
        if task_id in self.completed_tasks:
            return ("completed", self.completed_tasks[task_id])
        elif task_id in self.active_tasks:
            return ("processing", None)
        else:
            return ("not_found", None)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Отмена задачи.
        
        Args:
            task_id: ID задачи
            
        Returns:
            True если задача отменена, иначе False
        """
        # Удаление из активных задач
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            logger.info(f"Задача {task_id} отменена")
            return True
        
        # Задача не найдена
        return False
    
    def clear_completed_tasks(self):
        """Очистка списка выполненных задач."""
        self.completed_tasks.clear()
        logger.info("Список выполненных задач очищен")
    
    def shutdown(self):
        """Остановка всех рабочих потоков."""
        logger.info("Остановка прогрессивного загрузчика...")
        
        # Установка флага остановки
        self.stop_event.set()
        
        # Ожидание завершения рабочих потоков
        for worker in self.workers:
            worker.join(timeout=1.0)
        
        # Очистка очереди
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except:
                break
        
        logger.info("Прогрессивный загрузчик остановлен")


# Глобальный экземпляр прогрессивного загрузчика
progressive_loader = ProgressiveImageLoader()


def process_large_image_progressively(image_data: np.ndarray, operation: str, 
                                callback: Optional[Callable] = None,
                                chunk_size: int = 256) -> str:
    """
    Прогрессивная обработка большого изображения по частям.
    
    Args:
        image_data: Данные изображения
        operation: Тип операции ('window_level', 'downsample', 'segmentation')
        callback: Функция обратного вызова
        chunk_size: Размер блока обработки
        
    Returns:
        ID задачи
    """
    # Проверка размера изображения
    if image_data.nbytes < 50 * 1024 * 1024:  # Если меньше 50 МБ, обрабатываем целиком
        if operation == 'window_level':
            return progressive_loader.add_window_level_task(image_data, 0, 400, callback)
        elif operation == 'downsample':
            return progressive_loader.add_downsample_task(image_data, 0.5, callback)
        elif operation == 'segmentation':
            return progressive_loader.add_segmentation_task(image_data, 'U-Net', callback)
    
    # Разделение изображения на блоки
    height, width = image_data.shape[:2]
    task_ids = []
    
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            # Определение границ блока
            block_height = min(chunk_size, height - y)
            block_width = min(chunk_size, width - x)
            
            # Добавление задачи для блока
            if operation == 'window_level':
                task_id = progressive_loader.add_window_level_task(
                    image_data, 0, 400, callback, 
                    ProcessingPriority.NORMAL, (x, y, block_width, block_height)
                )
            elif operation == 'downsample':
                task_id = progressive_loader.add_downsample_task(
                    image_data, 0.5, callback,
                    ProcessingPriority.NORMAL, (x, y, block_width, block_height)
                )
            elif operation == 'segmentation':
                task_id = progressive_loader.add_segmentation_task(
                    image_data, 'U-Net', callback,
                    ProcessingPriority.NORMAL, (x, y, block_width, block_height)
                )
            
            task_ids.append(task_id)
    
    # Возвращаем ID первой задачи
    return task_ids[0] if task_ids else ""