from collections import OrderedDict
from typing import Dict, Any, Optional, Union
import numpy as np


class PreviewCache:
    """
    LRU кэш для хранения превью изображений
    """
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.cache = OrderedDict()
        
    def get(self, key: str) -> Optional[Union[np.ndarray, Dict]]:
        """Получить значение из кэша по ключу"""
        if key in self.cache:
            # Перемещаем элемент в конец (обновляем время использования)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None
    
    def put(self, key: str, value: Union[np.ndarray, Dict]) -> None:
        """Добавить значение в кэш"""
        if key in self.cache:
            # Обновляем существующий ключ
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Удаляем самый старый элемент (LRU)
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Очистить кэш"""
        self.cache.clear()
    
    def contains(self, key: str) -> bool:
        """Проверить наличие ключа в кэше"""
        return key in self.cache
    
    def size(self) -> int:
        """Получить текущий размер кэша"""
        return len(self.cache)
    
    def keys(self) -> list:
        """Получить список всех ключей в кэше"""
        return list(self.cache.keys())