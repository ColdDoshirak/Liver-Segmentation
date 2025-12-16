#!/usr/bin/env python3
"""
Скрипт для запуска GUI приложения LiverSeg 3D.

Этот скрипт предоставляет простой способ запуска приложения
с правильной настройкой окружения и обработкой ошибок.
"""

import sys
import os

# Добавление пути к модулю InferenceAnd3D в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Импорт QApplication из PyQt6
    from PyQt6.QtWidgets import QApplication
    from gui.main_window import MainWindow
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все необходимые зависимости установлены:")
    print("pip install PyQt6 pyvistaqt SimpleITK numpy torch torchvision ultralytics")
    sys.exit(1)

def main():
    """Основная функция запуска приложения."""
    # Создание приложения
    app = QApplication(sys.argv)
    
    # Установка стиля приложения
    app.setStyle('Fusion')
    
    # Создание и отображение главного окна
    window = MainWindow()
    window.show()
    
    # Запуск цикла обработки событий
    sys.exit(app.exec())

if __name__ == "__main__":
    main()