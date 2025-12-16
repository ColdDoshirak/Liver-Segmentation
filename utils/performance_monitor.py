"""
Модуль для мониторинга производительности операций обработки изображений.

Этот модуль предоставляет классы и функции для измерения времени выполнения
операций, выявления узких мест и логирования производительности.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from functools import wraps
from dataclasses import dataclass
from collections import defaultdict

# Настройка логирования
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Класс для хранения метрики производительности."""
    operation_name: str
    execution_time: float
    memory_usage: float  # в МБ
    data_size: tuple  # размер данных
    timestamp: float


class PerformanceMonitor:
    """
    Класс для мониторинга производительности операций обработки изображений.
    
    Позволяет отслеживать время выполнения операций, использование памяти
    и выявлять узкие места в производительности.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Инициализация монитора производительности.
        
        Args:
            max_history: Максимальное количество сохраняемых метрик
        """
        self.metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.max_history = max_history
        self.active_timers: Dict[str, float] = {}
        
    def start_timer(self, operation_name: str) -> None:
        """
        Запуск таймера для операции.
        
        Args:
            operation_name: Название операции
        """
        self.active_timers[operation_name] = time.time()
        
    def end_timer(self, operation_name: str, data_size: Optional[tuple] = None) -> float:
        """
        Остановка таймера и сохранение метрики.
        
        Args:
            operation_name: Название операции
            data_size: Размер обработанных данных
            
        Returns:
            Время выполнения операции в секундах
        """
        if operation_name not in self.active_timers:
            logger.warning(f"Таймер для операции '{operation_name}' не был запущен")
            return 0.0
            
        start_time = self.active_timers.pop(operation_name)
        execution_time = time.time() - start_time
        
        # Получение использования памяти
        memory_usage = self._get_memory_usage()
        
        # Создание метрики
        metric = PerformanceMetric(
            operation_name=operation_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            data_size=data_size or (0, 0, 0),
            timestamp=time.time()
        )
        
        # Сохранение метрики
        self.metrics[operation_name].append(metric)
        
        # Ограничение размера истории
        if len(self.metrics[operation_name]) > self.max_history:
            self.metrics[operation_name].pop(0)
        
        # Логирование
        logger.info(f"Операция '{operation_name}' выполнена за {execution_time:.3f}с, "
                   f"память: {memory_usage:.1f}МБ, размер данных: {data_size}")
        
        return execution_time
    
    def _get_memory_usage(self) -> float:
        """
        Получение текущего использования памяти.
        
        Returns:
            Использование памяти в МБ
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Конвертация в МБ
        except ImportError:
            # Если psutil недоступен, возвращаем 0
            return 0.0
    
    def get_average_time(self, operation_name: str) -> float:
        """
        Получение среднего времени выполнения операции.
        
        Args:
            operation_name: Название операции
            
        Returns:
            Среднее время выполнения в секундах
        """
        if operation_name not in self.metrics or not self.metrics[operation_name]:
            return 0.0
            
        times = [m.execution_time for m in self.metrics[operation_name]]
        return float(np.mean(times))
    
    def get_total_time(self, operation_name: str) -> float:
        """
        Получение общего времени выполнения операции.
        
        Args:
            operation_name: Название операции
            
        Returns:
            Общее время выполнения в секундах
        """
        if operation_name not in self.metrics or not self.metrics[operation_name]:
            return 0.0
            
        times = [m.execution_time for m in self.metrics[operation_name]]
        return sum(times)
    
    def get_slowest_operations(self, limit: int = 5) -> List[tuple]:
        """
        Получение списка самых медленных операций.
        
        Args:
            limit: Максимальное количество операций в списке
            
        Returns:
            Список кортежей (operation_name, average_time)
        """
        operation_times = []
        for operation_name in self.metrics:
            avg_time = self.get_average_time(operation_name)
            if avg_time > 0:
                operation_times.append((operation_name, avg_time))
        
        # Сортировка по убыванию времени
        operation_times.sort(key=lambda x: x[1], reverse=True)
        return operation_times[:limit]
    
    def get_memory_intensive_operations(self, limit: int = 5) -> List[tuple]:
        """
        Получение списка самых требовательных к памяти операций.
        
        Args:
            limit: Максимальное количество операций в списке
            
        Returns:
            Список кортежей (operation_name, average_memory)
        """
        operation_memory = []
        for operation_name in self.metrics:
            if self.metrics[operation_name]:
                avg_memory = np.mean([m.memory_usage for m in self.metrics[operation_name]])
                operation_memory.append((operation_name, avg_memory))
        
        # Сортировка по убыванию использования памяти
        operation_memory.sort(key=lambda x: x[1], reverse=True)
        return operation_memory[:limit]
    
    def clear_metrics(self, operation_name: Optional[str] = None) -> None:
        """
        Очистка метрик производительности.
        
        Args:
            operation_name: Название операции для очистки.
                         Если None, очищаются все метрики.
        """
        if operation_name:
            if operation_name in self.metrics:
                self.metrics[operation_name].clear()
                logger.info(f"Метрики для операции '{operation_name}' очищены")
        else:
            self.metrics.clear()
            logger.info("Все метрики производительности очищены")
    
    def generate_report(self) -> str:
        """
        Генерация отчета о производительности.
        
        Returns:
            Текстовый отчет о производительности
        """
        report = ["=== ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ ===\n"]
        
        # Самые медленные операции
        slow_ops = self.get_slowest_operations()
        if slow_ops:
            report.append("Самые медленные операции:")
            for i, (op_name, avg_time) in enumerate(slow_ops, 1):
                total_time = self.get_total_time(op_name)
                count = len(self.metrics[op_name])
                report.append(f"  {i}. {op_name}: {avg_time:.3f}с (среднее), "
                           f"{total_time:.3f}с (общее), {count} вызовов")
            report.append("")
        
        # Самые требовательные к памяти операции
        memory_ops = self.get_memory_intensive_operations()
        if memory_ops:
            report.append("Самые требовательные к памяти операции:")
            for i, (op_name, avg_memory) in enumerate(memory_ops, 1):
                report.append(f"  {i}. {op_name}: {avg_memory:.1f}МБ (среднее)")
            report.append("")
        
        # Общая статистика
        total_operations = sum(len(metrics) for metrics in self.metrics.values())
        report.append(f"Всего выполнено операций: {total_operations}")
        report.append(f"Отслеживаемых типов операций: {len(self.metrics)}")
        
        return "\n".join(report)


# Глобальный экземпляр монитора производительности
performance_monitor = PerformanceMonitor()


def performance_timer(operation_name: Optional[str] = None):
    """
    Декоратор для измерения времени выполнения функции.
    
    Args:
        operation_name: Название операции. Если None, используется имя функции.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Определение названия операции
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Запуск таймера
            performance_monitor.start_timer(op_name)
            
            try:
                # Выполнение функции
                result = func(*args, **kwargs)
                
                # Попытка получить размер данных из результата
                data_size = None
                if hasattr(result, 'shape'):
                    data_size = result.shape
                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    if hasattr(result[0], 'shape'):
                        data_size = result[0].shape
                
                # Остановка таймера
                performance_monitor.end_timer(op_name, data_size)
                
                return result
            except Exception as e:
                # Остановка таймера даже при ошибке
                performance_monitor.end_timer(op_name)
                raise e
        
        return wrapper
    return decorator


def log_performance_bottlenecks() -> None:
    """
    Логирование узких мест в производительности.
    """
    slow_ops = performance_monitor.get_slowest_operations(3)
    if slow_ops:
        logger.warning("Обнаружены узкие места в производительности:")
        for op_name, avg_time in slow_ops:
            if avg_time > 1.0:  # Если операция выполняется дольше 1 секунды
                logger.warning(f"  - {op_name}: {avg_time:.3f}с (среднее время)")
    
    memory_ops = performance_monitor.get_memory_intensive_operations(3)
    if memory_ops:
        logger.warning("Обнаружены операции с высоким потреблением памяти:")
        for op_name, avg_memory in memory_ops:
            if avg_memory > 500:  # Если операция использует более 500 МБ
                logger.warning(f"  - {op_name}: {avg_memory:.1f}МБ (среднее)")


def get_performance_summary() -> Dict[str, Any]:
    """
    Получение сводной информации о производительности.
    
    Returns:
        Словарь со сводной информацией
    """
    return {
        'total_operations': sum(len(metrics) for metrics in performance_monitor.metrics.values()),
        'operation_types': len(performance_monitor.metrics),
        'slowest_operations': performance_monitor.get_slowest_operations(5),
        'memory_intensive_operations': performance_monitor.get_memory_intensive_operations(5)
    }