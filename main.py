#!/usr/bin/env python3
"""
Главный файл запуска приложения LiverSeg 3D.

Этот модуль предоставляет основной вход в приложение с проверкой зависимостей,
обработкой исключений и информацией о версии.
"""

import sys
import os
import logging
from typing import List, Optional
# Добавление пути к модулю InferenceAnd3D в sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Версия приложения
__version__ = "1.0.0"
__app_name__ = "LiverSeg 3D"
__description__ = "Приложение для 3D сегментации печени на КТ изображениях"

# Настройка логирования с поддержкой кириллицы и Unicode
log_file_path = os.path.join(os.path.dirname(__file__), 'liverseg3d.log')
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[stream_handler, file_handler]
)
logger = logging.getLogger(__name__)


def check_python_version() -> bool:
    """
    Проверка версии Python.
    
    Returns:
        bool: True если версия Python поддерживается
    """
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        logger.error(f"Требуется Python {required_version[0]}.{required_version[1]} или выше. "
                    f"Текущая версия: {current_version[0]}.{current_version[1]}")
        return False
    
    logger.info(f"Версия Python: {sys.version}")
    return True


def check_dependencies() -> bool:
    """
    Проверка наличия необходимых зависимостей.
    
    Returns:
        bool: True если все зависимости установлены
    """
    required_packages = [
        'PyQt6',
        'pyvistaqt',
        'SimpleITK',
        'numpy',
        'torch',
        'torchvision',
        'ultralytics',
        'monai',
        'nnunetv2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"[OK] {package} установлен")
        except ImportError:
            logger.error(f"[X] {package} не установлен")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Отсутствуют следующие зависимости: {', '.join(missing_packages)}")
        logger.info("Установите зависимости с помощью команды: pip install -r requirements.txt")
        return False
    
    return True


def check_gpu_availability() -> bool:
    """
    Проверка доступности GPU.
    
    Returns:
        bool: True если GPU доступен
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Доступно GPU: {gpu_name} (всего: {gpu_count})")
            return True
        else:
            logger.warning("GPU не доступен. Будет использоваться CPU.")
            return False
    except Exception as e:
        logger.warning(f"Ошибка при проверке GPU: {e}")
        return False


def check_model_files() -> bool:
    """
    Проверка наличия файлов моделей.
    
    Returns:
        bool: True если все файлы моделей найдены
    """
    try:
        from config import validate_model_paths
        
        validation_results = validate_model_paths()
        all_models_available = all(validation_results.values())
        
        logger.info("Проверка файлов моделей:")
        for model_name, exists in validation_results.items():
            status = "[OK]" if exists else "[X]"
            logger.info(f"  {status} {model_name}: {'Доступна' if exists else 'Недоступна'}")
        
        if not all_models_available:
            logger.warning("Некоторые модели недоступны. Функциональность может быть ограничена.")
        
        return True  # Не блокируем запуск, если некоторые модели отсутствуют
        
    except Exception as e:
        logger.error(f"Ошибка при проверке файлов моделей: {e}")
        return False


def initialize_app() -> bool:
    """
    Инициализация приложения и проверка всех компонентов.
    
    Returns:
        bool: True если инициализация прошла успешно
    """
    logger.info(f"Запуск {__app_name__} v{__version__}")
    logger.info(f"Описание: {__description__}")
    
    # Проверка версии Python
    if not check_python_version():
        return False
    
    # Проверка зависимостей
    if not check_dependencies():
        return False
    
    # Проверка доступности GPU
    check_gpu_availability()
    
    # Проверка файлов моделей
    check_model_files()
    
    return True


def launch_gui():
    """
    Запуск GUI приложения.
    """
    try:
        try:
            # Импорт QApplication из PyQt6
            from PyQt6.QtWidgets import QApplication
            from gui.main_window import MainWindow
        except ImportError as e:
            logger.error(f"Ошибка импорта модулей GUI: {e}")
            logger.error("Попытка альтернативного импорта...")
            try:
                # Альтернативный способ импорта
                gui_path = os.path.join(os.path.dirname(__file__), 'gui')
                if gui_path not in sys.path:
                    sys.path.insert(0, gui_path)
                from PyQt6.QtWidgets import QApplication
                from main_window import MainWindow
                logger.info("Альтернативный импорт успешен")
            except ImportError as alt_error:
                logger.error(f"Альтернативный импорт также не удался: {alt_error}")
                logger.error("Убедитесь, что все необходимые зависимости установлены")
                raise
        
        # Создание приложения
        app = QApplication(sys.argv)
        
        # Установка стиля приложения
        app.setStyle('Fusion')
        
        # Установка информации о приложении
        app.setApplicationName(__app_name__)
        app.setApplicationVersion(__version__)
        app.setOrganizationName("LiverSeg Team")
        
        # Создание и отображение главного окна
        window = MainWindow()
        window.show()
        
        logger.info("GUI приложение успешно запущено")
        
        # Запуск цикла обработки событий
        sys.exit(app.exec())
        
    except ImportError as e:
        logger.error(f"Ошибка импорта модулей GUI: {e}")
        logger.error("Убедитесь, что все необходимые зависимости установлены")
        return False
    except Exception as e:
        logger.error(f"Ошибка при запуске GUI приложения: {e}")
        return False


def main():
    """
    Основная функция запуска приложения.
    """
    try:
        # Инициализация приложения
        if not initialize_app():
            logger.error("Инициализация приложения не удалась")
            sys.exit(1)
        
        # Запуск GUI
        launch_gui()
        
    except KeyboardInterrupt:
        logger.info("Приложение прервано пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()