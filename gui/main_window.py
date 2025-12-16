"""
Основное GUI приложение для сегментации печени на 3D КТ изображениях.

Этот модуль содержит класс MainWindow, который предоставляет пользовательский интерфейс
для загрузки DICOM данных, выполнения сегментации с использованием различных моделей
и визуализации результатов в 3D.
"""

import os
import sys
import numpy as np
import pyvista as pv
from typing import Optional, Dict, Any
import logging

# Настройка логирования с поддержкой кириллицы и Unicode
logger = logging.getLogger(__name__)

# Импорты PyQt6 с обработкой ошибок
try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
        QListWidget, QComboBox, QProgressBar, QTextEdit, QLabel,
        QMessageBox, QFileDialog, QSplitter, QGroupBox, QGridLayout,
        QSlider, QCheckBox, QSpinBox, QDoubleSpinBox, QScrollArea
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSlot, QSettings
    from PyQt6.QtGui import QFont
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
            QListWidget, QComboBox, QProgressBar, QTextEdit, QLabel,
            QMessageBox, QFileDialog, QSplitter, QGroupBox, QGridLayout,
            QSlider, QCheckBox, QSpinBox, QDoubleSpinBox
        )
        from PyQt5.QtCore import Qt, QThread, pyqtSlot, QSettings
        from PyQt5.QtGui import QFont
    except ImportError:
        raise ImportError("Требуется PyQt6 или PyQt5. Установите с помощью: pip install PyQt6")

# Импорт pyvistaqt для 3D визуализации
try:
    from pyvistaqt import QtInteractor
except ImportError:
    raise ImportError("Требуется pyvistaqt. Установите с помощью: pip install pyvistaqt")

# Импорты из локальных модулей с улучшенной обработкой ошибок
try:
    # Первая попытка: абсолютные импорты от корня проекта
    from InferenceAnd3D.gui.workers import DicomLoaderWorker, SegmentationWorker, VolumeCalculationWorker, DisplayUpdateWorker
    from InferenceAnd3D.utils.dicom_loader import DicomLoader
    from InferenceAnd3D.config import get_model_path, validate_model_paths, get_config
    from InferenceAnd3D.utils.performance_monitor import performance_monitor, log_performance_bottlenecks
    from InferenceAnd3D.utils.progressive_loader import progressive_loader, process_large_image_progressively
    logger.info("Модули успешно импортированы с абсолютными путями от корня проекта")
except ImportError as import_error:
    logger.warning(f"Не удалось импортировать модули с абсолютными путями от корня: {import_error}")
    try:
        # Вторая попытка: абсолютные импорты (предполагается, что корневой путь уже добавлен в sys.path)
        from gui.workers import DicomLoaderWorker, SegmentationWorker, VolumeCalculationWorker, DisplayUpdateWorker
        from utils.dicom_loader import DicomLoader
        from config import get_model_path, validate_model_paths, get_config
        from utils.performance_monitor import performance_monitor, log_performance_bottlenecks
        from utils.progressive_loader import progressive_loader, process_large_image_progressively
        logger.info("Модули успешно импортированы с абсолютными путями")
    except ImportError as second_error:
        logger.error(f"Критическая ошибка импорта модулей: {second_error}")
        raise ImportError(f"Не удалось импортировать необходимые модули. Проверьте структуру проекта: {second_error}")

# Настройка логирования (уже настроено выше)
logging.basicConfig(level=logging.INFO)


class MainWindow(QMainWindow):
    """
    Главное окно приложения для сегментации печени.
    
    Предоставляет интерфейс для загрузки DICOM данных, выбора модели сегментации,
    запуска процесса сегментации и визуализации результатов в 3D.
    """
    
    def __init__(self):
        """Инициализация главного окна приложения."""
        super().__init__()
        
        # Инициализация переменных
        self.dicom_folder = ""
        self.current_series_id = ""
        self.current_image_data = None
        self.current_mask_data = None
        self.current_voxel_volume = 1.0
        self.liver_volume_ml = 0.0
        
        # Рабочие потоки
        self.dicom_loader_worker = None
        self.segmentation_worker = None
        self.volume_calculation_worker = None
        self.display_update_worker = None
        
        # Механизм отложенного обновления (debounce)
        self._update_timer = None
        self._pending_update = False
        
        # Настройки контраста и видимости
        self.window_center = 0  # Центр окна контраста
        self.window_width = 400  # Ширина окна контраста
        self.show_ct_image = True  # Видимость КТ изображения
        self.show_mask = True  # Видимость маски
        self.mask_opacity = 0.5  # Прозрачность маски
        
        # Настройки 2D режима
        self.view_mode = "3D"  # Режим просмотра: "3D" или "2D"
        self.current_slice = 0  # Текущий срез в 2D режиме
        self.max_slices = 1  # Максимальное количество срезов
        
        # Настройки приложения
        self.settings = QSettings("LiverSeg3D", "Settings")
        # Инициализация PreviewManager
        from InferenceAnd3D.utils.preview_manager import PreviewManager
        from InferenceAnd3D.config import get_config
        preview_config = get_config()['preview']
        self.preview_manager = PreviewManager(cache_size=preview_config['cache_size'])
        self.preview_manager.set_resolution(preview_config['default_resolution'])
        self.preview_manager.update_use_preview_setting(preview_config['use_preview_for_display'])
        
        # Инициализация переменных для отображения выделенной области
        self.show_masked_region = False
        self.mask_display_mode = 'multiply'
        
        # Инициализация UI (включая создание logs_text)
        self.init_ui()
        
        # Загрузка сохраненных настроек ПОСЛЕ инициализации UI элементов
        self.load_settings()
        
        # Настройка PyVista для обработки пустых mesh
        try:
            pv.global_theme.allow_empty_mesh = True
            self.log_message("PyVista настроен для обработки пустых mesh")
        except Exception as e:
            self.log_message(f"Предупреждение: не удалось настроить PyVista: {str(e)}")
        
        # Проверка путей к моделям
        self.validate_models()
        
        # Настройка логирования
        self.log_message("Приложение LiverSeg 3D запущено")
        
    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        # Настройка главного окна
        self.setWindowTitle("LiverSeg 3D - Сегментация печени")
        self.setGeometry(100, 100, 1200, 800)
        
        # Создание центрального виджета
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Создание основного горизонтального разделителя
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        central_widget_layout = QHBoxLayout(central_widget)
        central_widget_layout.addWidget(main_splitter)
        central_widget.setLayout(central_widget_layout)
        
        # Создание левой панели (настройки)
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Создание центральной панели (3D визуализация)
        central_panel = self.create_central_panel()
        main_splitter.addWidget(central_panel)
        
        # Установка пропорций разделителя
        main_splitter.setSizes([400, 800])
        
    def create_left_panel(self) -> QWidget:
        """
        Создание левой панели с элементами управления.
        
        Returns:
            QWidget: Левая панель с элементами управления
        """
        # Создаем прокручиваемую область
        scroll_area = QScrollArea()
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Группа загрузки DICOM
        dicom_group = QGroupBox("Загрузка DICOM")
        dicom_layout = QVBoxLayout(dicom_group)
        
        # Кнопка выбора папки
        self.select_folder_btn = QPushButton("Select DICOM Folder")
        self.select_folder_btn.clicked.connect(self.select_dicom_folder)
        dicom_layout.addWidget(self.select_folder_btn)
        
        # Список DICOM серий
        self.series_list = QListWidget()
        self.series_list.itemSelectionChanged.connect(self.on_series_selected)
        dicom_layout.addWidget(QLabel("DICOM серии:"))
        dicom_layout.addWidget(self.series_list)
        
        scroll_layout.addWidget(dicom_group)
        
        # Группа настроек сегментации
        segmentation_group = QGroupBox("Настройки сегментации")
        segmentation_layout = QVBoxLayout(segmentation_group)
        
        # Выбор модели
        segmentation_layout.addWidget(QLabel("Модель:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv11", "U-Net", "nnU-Net"])
        segmentation_layout.addWidget(self.model_combo)
        
        # Кнопка запуска сегментации
        self.start_segmentation_btn = QPushButton("Start Segmentation")
        self.start_segmentation_btn.clicked.connect(self.start_segmentation)
        self.start_segmentation_btn.setEnabled(False)
        segmentation_layout.addWidget(self.start_segmentation_btn)
        
        scroll_layout.addWidget(segmentation_group)
        
        # Группа настроек превью
        preview_group = QGroupBox("Настройки превью")
        preview_layout = QVBoxLayout(preview_group)
        
        # Выбор разрешения превью
        preview_layout.addWidget(QLabel("Разрешение превью:"))
        self.preview_resolution_combo = QComboBox()
        self.preview_resolution_combo.addItems(["Оригинал", "128", "256", "512"])
        self.preview_resolution_combo.currentTextChanged.connect(self.on_preview_resolution_changed)
        preview_layout.addWidget(self.preview_resolution_combo)
        
        # Чекбокс использования превью для отображения
        self.use_preview_checkbox = QCheckBox("Использовать превью для отображения")
        self.use_preview_checkbox.setChecked(True)
        self.use_preview_checkbox.stateChanged.connect(self.on_use_preview_changed)
        preview_layout.addWidget(self.use_preview_checkbox)
        
        # Кнопка обновления превью
        self.update_preview_btn = QPushButton("Обновить превью")
        self.update_preview_btn.clicked.connect(self.on_update_preview_clicked)
        preview_layout.addWidget(self.update_preview_btn)
        
        scroll_layout.addWidget(preview_group)
        
        # Группа настроек контраста и видимости
        visualization_group = QGroupBox("Настройки визуализации")
        visualization_layout = QVBoxLayout(visualization_group)
        
        # Подгруппа настроек контраста
        contrast_group = QGroupBox("Настройки контраста КТ")
        contrast_layout = QGridLayout(contrast_group)
        
        # Window Center
        contrast_layout.addWidget(QLabel("Центр окна:"), 0, 0)
        self.window_center_spin = QSpinBox()
        self.window_center_spin.setRange(-1000, 100)
        self.window_center_spin.setValue(self.window_center)
        self.window_center_spin.valueChanged.connect(self.on_window_center_changed)
        contrast_layout.addWidget(self.window_center_spin, 0, 1)
        
        self.window_center_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_center_slider.setRange(-1000, 1000)
        self.window_center_slider.setValue(self.window_center)
        self.window_center_slider.valueChanged.connect(self.on_window_center_changed)
        contrast_layout.addWidget(self.window_center_slider, 0, 2)
        
        # Window Width
        contrast_layout.addWidget(QLabel("Ширина окна:"), 1, 0)
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(1, 4000)
        self.window_width_spin.setValue(self.window_width)
        self.window_width_spin.valueChanged.connect(self.on_window_width_changed)
        contrast_layout.addWidget(self.window_width_spin, 1, 1)
        
        self.window_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_width_slider.setRange(1, 4000)
        self.window_width_slider.setValue(self.window_width)
        self.window_width_slider.valueChanged.connect(self.on_window_width_changed)
        contrast_layout.addWidget(self.window_width_slider, 1, 2)
        
        visualization_layout.addWidget(contrast_group)
        
        # Подгруппа управления видимостью слоев
        layers_group = QGroupBox("Управление слоями")
        layers_layout = QGridLayout(layers_group)
        
        # Чекбоксы видимости
        self.show_ct_checkbox = QCheckBox("КТ изображение")
        self.show_ct_checkbox.setChecked(self.show_ct_image)
        self.show_ct_checkbox.stateChanged.connect(self.on_layer_visibility_changed)
        layers_layout.addWidget(self.show_ct_checkbox, 0, 0)
        
        self.show_mask_checkbox = QCheckBox("Маска сегментации")
        self.show_mask_checkbox.setChecked(self.show_mask)
        self.show_mask_checkbox.stateChanged.connect(self.on_layer_visibility_changed)
        layers_layout.addWidget(self.show_mask_checkbox, 0, 1)
        
        # Чекбокс для показа выделенной области
        self.show_masked_region_checkbox = QCheckBox("Показать выделенную область")
        self.show_masked_region_checkbox.setChecked(False)
        self.show_masked_region_checkbox.stateChanged.connect(self.on_masked_region_visibility_changed)
        layers_layout.addWidget(self.show_masked_region_checkbox, 0, 2)
        
        # Прозрачность маски
        layers_layout.addWidget(QLabel("Прозрачность маски:"), 1, 0)
        self.mask_opacity_spin = QDoubleSpinBox()
        self.mask_opacity_spin.setRange(0.0, 1.0)
        self.mask_opacity_spin.setSingleStep(0.1)
        self.mask_opacity_spin.setValue(self.mask_opacity)
        self.mask_opacity_spin.valueChanged.connect(self.on_mask_opacity_changed)
        layers_layout.addWidget(self.mask_opacity_spin, 1, 1)
        
        self.mask_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_opacity_slider.setRange(0, 100)
        self.mask_opacity_slider.setValue(int(self.mask_opacity * 100))
        self.mask_opacity_slider.valueChanged.connect(self.on_mask_opacity_slider_changed)
        layers_layout.addWidget(self.mask_opacity_slider, 1, 2)
        
        # Выбор режима отображения выделенной области
        layers_layout.addWidget(QLabel("Режим отображения:"), 2, 0)
        self.mask_display_mode_combo = QComboBox()
        self.mask_display_mode_combo.addItems(["Контур", "Полупрозрачное наложение", "Только печень"])
        self.mask_display_mode_combo.currentTextChanged.connect(self.on_mask_display_mode_changed)
        layers_layout.addWidget(self.mask_display_mode_combo, 2, 1)
        
        visualization_layout.addWidget(layers_group)
        
        # Группа режима просмотра
        view_mode_group = QGroupBox("Режим просмотра")
        view_mode_layout = QVBoxLayout(view_mode_group)
        
        # Переключатель режима просмотра
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["3D режим", "2D режим"])
        self.view_mode_combo.currentTextChanged.connect(self.on_view_mode_changed)
        view_mode_layout.addWidget(self.view_mode_combo)
        
        # Группа навигации по срезам (изначально скрыта)
        self.slice_navigation_group = QGroupBox("Навигация по срезам")
        slice_navigation_layout = QVBoxLayout(self.slice_navigation_group)
        
        # Слайдер для навигации по срезам
        slice_slider_layout = QHBoxLayout()
        slice_slider_layout.addWidget(QLabel("Срез:"))
        
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        slice_slider_layout.addWidget(self.slice_slider)
        
        self.slice_label = QLabel("0 / 100")
        slice_slider_layout.addWidget(self.slice_label)
        
        slice_navigation_layout.addLayout(slice_slider_layout)
        
        # Кнопки быстрой навигации
        slice_buttons_layout = QHBoxLayout()
        
        self.first_slice_btn = QPushButton("Первый")
        self.first_slice_btn.clicked.connect(lambda: self.set_slice(0))
        slice_buttons_layout.addWidget(self.first_slice_btn)
        
        self.prev_slice_btn = QPushButton("Предыдущий")
        self.prev_slice_btn.clicked.connect(lambda: self.set_slice(self.current_slice - 1))
        slice_buttons_layout.addWidget(self.prev_slice_btn)
        
        self.next_slice_btn = QPushButton("Следующий")
        self.next_slice_btn.clicked.connect(lambda: self.set_slice(self.current_slice + 1))
        slice_buttons_layout.addWidget(self.next_slice_btn)
        
        self.last_slice_btn = QPushButton("Последний")
        self.last_slice_btn.clicked.connect(lambda: self.set_slice(self.max_slices - 1))
        slice_buttons_layout.addWidget(self.last_slice_btn)
        
        slice_navigation_layout.addLayout(slice_buttons_layout)
        
        # Изначально скрываем навигацию по срезам
        self.slice_navigation_group.setVisible(False)
        
        view_mode_layout.addWidget(self.slice_navigation_group)
        
        visualization_layout.addWidget(view_mode_group)
        
        scroll_layout.addWidget(visualization_group)
        
        # Группа прогресса
        progress_group = QGroupBox("Прогресс")
        progress_layout = QVBoxLayout(progress_group)
        
        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        scroll_layout.addWidget(progress_group)
        
        # Группа логов
        logs_group = QGroupBox("Логи и статус")
        logs_layout = QVBoxLayout(logs_group)
        
        # Текстовое поле для логов
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setMaximumHeight(150)
        logs_layout.addWidget(self.logs_text)
        
        scroll_layout.addWidget(logs_group)
        
        # Группа результатов
        results_group = QGroupBox("Результаты")
        results_layout = QVBoxLayout(results_group)
        
        # Метка объема печени
        self.volume_label = QLabel("Liver Volume: ... ml")
        self.volume_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        results_layout.addWidget(self.volume_label)
        
        scroll_layout.addWidget(results_group)
        
        # Группа производительности
        performance_group = QGroupBox("Производительность")
        performance_layout = QVBoxLayout(performance_group)
        
        # Кнопка отчета о производительности
        self.performance_report_btn = QPushButton("Отчет о производительности")
        self.performance_report_btn.clicked.connect(self.show_performance_report)
        performance_layout.addWidget(self.performance_report_btn)
        
        # Кнопка очистки метрик
        self.clear_metrics_btn = QPushButton("Очистить метрики")
        self.clear_metrics_btn.clicked.connect(self.clear_performance_metrics)
        performance_layout.addWidget(self.clear_metrics_btn)
        
        scroll_layout.addWidget(performance_group)
        
        # Добавление растягивающегося пространства
        scroll_layout.addStretch()
        
        # Настройка прокручиваемой области
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        
        # Создаем основной виджет и добавляем в него прокручиваемую область
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(scroll_area)
        
        return left_panel
        
    def create_central_panel(self) -> QWidget:
        """
        Создание центральной панели с 3D/2D визуализацией.
        
        Returns:
            QWidget: Центральная панель с визуализацией
        """
        central_panel = QWidget()
        central_layout = QVBoxLayout(central_panel)
        
        # Создание виджета для 3D визуализации
        self.plotter = QtInteractor(self)
        central_layout.addWidget(self.plotter.interactor)
        
        # Создание виджета для 2D визуализации (изначально скрыто)
        from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setVisible(False)
        central_layout.addWidget(self.graphics_view)
        
        # Настройка начального вида
        self.setup_initial_view()
        
        return central_panel
        
    def setup_initial_view(self):
        """Настройка начального вида 3D визуализации."""
        # Очистка сцены
        self.plotter.clear()
        
        # Добавление координатных осей
        self.plotter.add_axes()
        
        # Попытка установить фон (с обработкой возможной ошибки)
        try:
            self.plotter.background_color = "white"
        except Exception as e:
            logger.warning(f"Не удалось установить цвет фона: {str(e)}")
        
        # Обновление сцены
        self.plotter.show()
        
    @pyqtSlot()
    def select_dicom_folder(self):
        """Обработчик нажатия кнопки выбора папки с DICOM файлами."""
        # Открытие диалога выбора папки
        folder_path = QFileDialog.getExistingDirectory(
            self, "Выберите папку с DICOM файлами", ""
        )
        
        if folder_path:
            self.dicom_folder = folder_path
            self.log_message(f"Выбрана папка: {folder_path}")
            
            # Загрузка списка DICOM серий
            self.load_dicom_series()
            
    def load_dicom_series(self):
        """Загрузка списка DICOM серий из выбранной папки."""
        try:
            # Очистка списка серий
            self.series_list.clear()
            
            # Создание рабочего потока для загрузки DICOM серий
            self.dicom_loader_worker = DicomLoaderWorker()
            self.dicom_loader_worker.progress_updated.connect(self.update_progress)
            self.dicom_loader_worker.error_occurred.connect(self.show_error)
            
            # Получение списка серий напрямую (без потока для простоты)
            dicom_loader = DicomLoader()
            series_ids = dicom_loader.get_dicom_series(self.dicom_folder)
            
            # Добавление серий в список
            for series_id in series_ids:
                # Получение информации о серии
                try:
                    series_info = dicom_loader.get_series_info(self.dicom_folder, series_id)
                    item_text = f"Серия {series_id} ({series_info['num_slices']} срезов)"
                    self.series_list.addItem(item_text)
                    # Сохранение ID серии в качестве данных элемента
                    item = self.series_list.item(self.series_list.count() - 1)
                    if item:
                        item.setData(Qt.ItemDataRole.UserRole, series_id)
                except Exception as e:
                    self.log_message(f"Ошибка при получении информации о серии {series_id}: {str(e)}")
                    self.series_list.addItem(f"Серия {series_id} (ошибка)")
                    item = self.series_list.item(self.series_list.count() - 1)
                    if item:
                        item.setData(Qt.ItemDataRole.UserRole, series_id)
            
            self.log_message(f"Найдено {len(series_ids)} DICOM серий")
            
        except Exception as e:
            self.show_error(f"Ошибка при загрузке DICOM серий: {str(e)}")
            
    @pyqtSlot()
    def on_series_selected(self):
        """Обработчик выбора DICOM серии из списка."""
        # Получение выбранного элемента
        current_item = self.series_list.currentItem()
        if not current_item:
            return
            
        # Получение ID серии
        self.current_series_id = current_item.data(Qt.ItemDataRole.UserRole)
        if not self.current_series_id:
            return
            
        self.log_message(f"Выбрана серия: {self.current_series_id}")
        
        # Активация кнопки сегментации
        self.start_segmentation_btn.setEnabled(True)
        
        # Загрузка и отображение серии
        self.load_and_display_series()
        
    def load_and_display_series(self):
        """Загрузка и отображение выбранной DICOM серии."""
        try:
            # Создание рабочего потока для загрузки DICOM серии
            self.dicom_loader_worker = DicomLoaderWorker()
            self.dicom_loader_worker.progress_updated.connect(self.update_progress)
            self.dicom_loader_worker.series_loaded.connect(self.on_series_loaded)
            self.dicom_loader_worker.error_occurred.connect(self.show_error)
            
            # Установка параметров и запуск потока
            self.dicom_loader_worker.load_series(self.dicom_folder, self.current_series_id)
            self.dicom_loader_worker.start()
            
            self.log_message("Загрузка DICOM серии...")
            
        except Exception as e:
            self.show_error(f"Ошибка при загрузке DICOM серии: {str(e)}")
            
    @pyqtSlot(dict)
    def on_series_loaded(self, series_data: Dict[str, Any]):
        """
        Обработчик завершения загрузки DICOM серии.
        
        Args:
            series_data: Словарь с данными загруженной серии
        """
        try:
            # Сохранение данных
            self.current_image_data = series_data['image_array']
            self.current_voxel_volume = series_data['voxel_volume']
            
            # Обновление информации о срезах для 2D режима
            if len(self.current_image_data.shape) == 3:
                self.max_slices = self.current_image_data.shape[0]
                self.slice_slider.setRange(0, self.max_slices - 1)
                self.slice_slider.setValue(0)
                self.current_slice = 0
                self.update_slice_label()
            
            # Обновление отображения с учетом настроек
            self.update_display()
            
            self.log_message(f"DICOM серия успешно загружена. Размер: {self.current_image_data.shape}")
            
        except Exception as e:
            self.show_error(f"Ошибка при обработке загруженной серии: {str(e)}")
            
    def display_volume(self, volume_data: np.ndarray):
        """
        Оптимизированное отображение 3D объема в визуализаторе с применением настроек контраста.
        
        Args:
            volume_data: 3D массив данных объема
        """
        try:
            # Применение настроек окна/уровня
            windowed_volume = self.apply_window_level(volume_data)
            
            # Создание 3D объема для визуализации с оптимизацией
            grid = pv.ImageData()
            grid.dimensions = np.array(volume_data.shape) + 1
            grid.spacing = (1.0, 1.0, 1.0)  # Размер вокселя
            grid.origin = (0, 0, 0)
            
            # Оптимизированное присвоение данных - используем ravel() вместо flatten()
            # ravel() создает представление вместо копии, что эффективнее для больших массивов
            grid.cell_data["values"] = windowed_volume.ravel(order="F")
            
            # Отображение объема с использованием volume rendering
            self.plotter.add_volume(grid, cmap="gray", opacity="sigmoid", name="volume")
            
        except Exception as e:
            self.show_error(f"Ошибка при отображении объема: {str(e)}")
            
    @pyqtSlot()
    def start_segmentation(self):
        """Обработчик нажатия кнопки запуска сегментации."""
        if self.current_image_data is None:
            self.show_error("Сначала загрузите DICOM серию")
            return
            
        try:
            # Получение выбранной модели
            model_type = self.model_combo.currentText()
            
            # Получение пути к модели из конфигурации
            model_path = get_model_path(model_type)
            
            if model_path is None or not os.path.exists(model_path):
                self.show_error(f"Файл модели не найден: {model_path}")
                return
                
            # Создание рабочего потока для сегментации
            self.segmentation_worker = SegmentationWorker()
            self.segmentation_worker.progress_updated.connect(self.update_progress)
            self.segmentation_worker.segmentation_completed.connect(self.on_segmentation_completed)
            self.segmentation_worker.volume_calculated.connect(self.on_volume_calculated)
            self.segmentation_worker.error_occurred.connect(self.show_error)
            
            # Установка параметров и запуск потока
            self.segmentation_worker.segment_image(self.current_image_data, model_type, model_path)
            self.segmentation_worker.start()
            
            self.log_message(f"Запущена сегментация с использованием модели: {model_type}")
            
            # Блокировка кнопки на время выполнения
            self.start_segmentation_btn.setEnabled(False)
            
        except Exception as e:
            self.show_error(f"Ошибка при запуске сегментации: {str(e)}")
            self.start_segmentation_btn.setEnabled(True)
            
    def validate_models(self):
        """Проверка доступности моделей и вывод информации в лог."""
        validation_results = validate_model_paths()
        
        # Проверяем, инициализирован ли UI перед вызовом log_message
        if hasattr(self, 'logs_text'):
            self.log_message("Проверка доступности моделей:")
            for model_name, exists in validation_results.items():
                status = "[OK]" if exists else "[X]"
                self.log_message(f"  {status} {model_name}: {'Доступна' if exists else 'Недоступна'}")
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Получение информации о конфигурации.
        
        Returns:
            Словарь с информацией о конфигурации
        """
        return get_config()
            
    @pyqtSlot(object)
    def on_segmentation_completed(self, result: object):
        """
        Обработчик завершения сегментации.

        Args:
            result: Маска сегментации или словарь с маской и типом модели
        """
        try:
            # Проверяем тип результата - может быть маской или словарем с маской и типом модели
            if isinstance(result, dict):
                mask_data = result.get('mask')
                model_type = result.get('model_type', '')
            else:
                mask_data = result
                model_type = ''  # Неизвестный тип модели
            
            # Сохранение маски
            self.current_mask_data = mask_data
            
            # Постобработка маски с учетом размеров изображения и типа модели
            if self.current_image_data is not None:
                from InferenceAnd3D.utils.dicom_loader import postprocess_mask
                self.current_mask_data = postprocess_mask(mask_data, self.current_image_data.shape, model_type=model_type)
            
            # Если используется отображение превью, масштабируем маску под размер превью
            if self.preview_manager.use_preview_for_display and self.current_image_data is not None:
                # Получаем размер превью для текущего разрешения
                preview_shape = self.preview_manager.get_preview_shape_for_resolution(
                    self.current_image_data.shape,
                    self.preview_manager.current_resolution
                )
                
                # Масштабируем маску под размер превью
                from InferenceAnd3D.utils.mask_scaler import MaskScaler
                mask_scaler = MaskScaler()
                self.current_mask_data = mask_scaler.scale_mask(self.current_mask_data, preview_shape)
            
            # Сохранение типа модели для последующего использования
            self.current_model_type = model_type
            # Обновление отображения с учетом настроек
            self.update_display(model_type=model_type)
            
            self.log_message("Сегментация завершена")
            
            # Расчет объема
            self.calculate_liver_volume(self.current_mask_data)
            
        except Exception as e:
            self.show_error(f"Ошибка при обработке результатов сегментации: {str(e)}")
        finally:
            # Разблокировка кнопки
            self.start_segmentation_btn.setEnabled(True)
            
    def display_mask(self, mask_data: np.ndarray, model_type: str = "Unknown"):
        """
        Отображение маски сегментации в 3D визуализаторе с учетом настроек прозрачности и типа модели.

        Args:
            mask_data: Маска сегментации
            model_type: Тип модели для цветовой дифференциации
        """
        try:
            # Проверка на пустую маску
            if mask_data is None or mask_data.size == 0:
                self.log_message("Предупреждение: получена пустая маска")
                return
                
            # Проверка, содержит ли маска какие-либо значения > 0
            if np.sum(mask_data > 0) == 0:
                self.log_message("Предупреждение: маска не содержит сегментированных объектов (все значения равны 0)")
                return
                
            # Логирование информации о маске
            self.log_message(f"Отображение маски: форма={mask_data.shape}, "
                            f"мин={mask_data.min():.3f}, макс={mask_data.max():.3f}, "
                            f"ненулевых вокселей={np.sum(mask_data > 0)}")
            
            # Убедимся, что маска бинарная (0 или 1)
            if mask_data.max() > 1:
                mask_data = (mask_data > 0.5).astype(np.uint8)
            
            # Проверка и выравнивание размеров маски изображения
            if self.current_image_data is not None:
                mask_data = self._ensure_mask_size_match(mask_data, self.current_image_data.shape)
            
            # Применяем морфологические операции для очистки маски от шумов
            mask_data = self._clean_mask(mask_data)
            
            # Создание 3D объема для визуализации маски с правильными параметрами
            grid = pv.ImageData()
            grid.dimensions = np.array(mask_data.shape) + 1
            grid.spacing = (1.0, 1.0, 1.0)
            grid.origin = (0, 0, 0)
            
            # Используем cell_data для соответствия размеру маски
            grid.cell_data["mask"] = mask_data.ravel(order="F")
            # Конвертируем cell_data в point_data для использования contour
            grid = grid.cell_data_to_point_data()
            
            # Извлечение поверхности с пороговым значением 0.5 (для бинарной маски)
            try:
                # Используем метод contour для извлечения поверхности
                # Устанавливаем правильный метод для point_data
                liver_surface = grid.contour(scalars="mask", isosurfaces=[0.5])
                
                # Проверка, что поверхность не пустая
                if liver_surface.n_points == 0:
                    self.log_message("Предупреждение: не удалось создать поверхность из маски (пустой mesh)")
                    # Пробуем альтернативный метод
                    self._display_mask_as_volume(mask_data)
                    return
                
                # Сглаживание поверхности для более реалистичного вида
                try:
                    liver_surface = liver_surface.smooth(n_iter=20, relaxation_factor=0.1)
                except:
                    pass  # Если сглаживание не удалось, продолжаем без него
                
                # Определение цвета в зависимости от типа модели
                # Обеспечиваем, что model_type не является None и безопасно вызываем lower()
                safe_model_type = model_type if model_type is not None else "Unknown"
                if safe_model_type.lower() in ['yolo', 'yolov1', 'yolov11']:
                    color = "blue"
                elif safe_model_type.lower() in ['u-net', 'unet']:
                    color = "red"
                elif safe_model_type.lower() in ['nnunet', 'nn-u-net', 'nnu-net']:
                    color = "green"
                else:
                    color = "red" # Цвет по умолчанию
                
                # Отображение поверхности печени с текущей прозрачностью и цветом
                self.plotter.add_mesh(
                    liver_surface,
                    color=color,
                    opacity=self.mask_opacity,
                    name="liver_mask",
                    style='surface',
                    smooth_shading=True
                )
                
                self.log_message(f"Успешно создана 3D поверхность печени: {liver_surface.n_points} точек, {liver_surface.n_cells} ячеек")
                
            except Exception as contour_error:
                self.log_message(f"Ошибка при создании контура: {str(contour_error)}")
                # Альтернативный метод - отображение вокселей
                self._display_mask_as_volume(mask_data)
            
        except Exception as e:
            self.show_error(f"Ошибка при отображении маски: {str(e)}")
    
    def _ensure_mask_size_match(self, mask_data: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Обеспечение соответствия размеров маски и изображения.
        
        Args:
            mask_data: Маска сегментации
            target_shape: Целевая форма (размеры изображения)
            
        Returns:
            Маска с размерами, соответствующими изображению
        """
        try:
            # Проверяем текущие размеры маски
            current_shape = mask_data.shape
            
            # Если размеры уже совпадают, возвращаем маску без изменений
            if current_shape == target_shape:
                self.log_message(f"Размеры маски уже совпадают с изображением: {current_shape}")
                return mask_data
            
            self.log_message(f"Выравнивание размеров маски: с {current_shape} до {target_shape}")
            
            # Используем универсальный подход с изменением размерности
            try:
                from scipy import ndimage
                
                # Обработка изменения размерности
                if len(mask_data.shape) != len(target_shape):
                    # Если размерности не совпадают, создаем новую маску нужной размерности
                    if len(mask_data.shape) < len(target_shape):
                        # Добавляем измерения
                        new_shape = [1] * (len(target_shape) - len(mask_data.shape)) + list(mask_data.shape)
                        expanded_mask = np.zeros(new_shape, dtype=mask_data.dtype)
                        # Копируем данные в центр новых измерений
                        slices = tuple([slice(0, s) for s in mask_data.shape])
                        expanded_mask[(slice(None),) * (len(target_shape) - len(mask_data.shape)) + slices] = mask_data
                        mask_data = expanded_mask
                    else:
                        # Уменьшаем размерность, проецируя на первые измерения
                        # Берем максимальное значение по дополнительным измерениям
                        while len(mask_data.shape) > len(target_shape):
                            mask_data = np.max(mask_data, axis=0)
                
                # Вычисление коэффициентов масштабирования
                zoom_factors = [target / current for target, current in zip(target_shape, mask_data.shape)]
                
                # Масштабирование маски
                scaled_mask = ndimage.zoom(mask_data, zoom_factors, order=0)  # order=0 для nearest-neighbor
                
                # Бинаризация результата
                scaled_mask = (scaled_mask > 0.5).astype(mask_data.dtype)
                
                self.log_message(f"Маска успешно масштабирована с коэффициентами {zoom_factors}")
                return scaled_mask
                
            except ImportError:
                # Если scipy недоступен, используем упрощенный подход
                self.log_message("Предупреждение: scipy недоступен, используется упрощенное выравнивание")
                
                # Если маска 2D, а изображение 3D
                if len(current_shape) == 2 and len(target_shape) == 3:
                    # Создаем 3D маску, повторяя 2D маску для каждого среза
                    mask_3d = np.zeros(target_shape, dtype=mask_data.dtype)
                    
                    # Определяем, как разместить 2D маску в 3D пространстве
                    if current_shape[0] == target_shape[1] and current_shape[1] == target_shape[2]:
                        # Размеры совпадают с срезами изображения, размещаем в центральном срезе
                        center_slice = target_shape[0] // 2
                        mask_3d[center_slice, :, :] = mask_data
                        self.log_message(f"2D маска размещена в центральном срезе {center_slice}")
                    else:
                        # Просто обрезаем/дополняем
                        min_y = min(current_shape[0], target_shape[1])
                        min_x = min(current_shape[1], target_shape[2])
                        center_slice = target_shape[0] // 2
                        mask_3d[center_slice, :min_y, :min_x] = mask_data[:min_y, :min_x]
                        self.log_message(f"2D маска обрезана и размещена в центральном срезе {center_slice}")
                    
                    return mask_3d
                
                # Если маска 3D, а изображение 2D
                elif len(current_shape) == 3 and len(target_shape) == 2:
                    # Берем центральный срез 3D маски
                    center_slice = current_shape[0] // 2
                    mask_2d = mask_data[center_slice, :, :]
                    
                    # Обрезаем/дополняем при необходимости
                    if mask_2d.shape != target_shape:
                        min_y = min(mask_2d.shape[0], target_shape[0])
                        min_x = min(mask_2d.shape[1], target_shape[1])
                        result = np.zeros(target_shape, dtype=mask_data.dtype)
                        result[:min_y, :min_x] = mask_2d[:min_y, :min_x]
                        mask_2d = result
                        self.log_message(f"3D маска преобразована в 2D и обрезана")
                    
                    return mask_2d
                
                # Если обе маски 3D, но размеры не совпадают
                elif len(current_shape) == 3 and len(target_shape) == 3:
                    # Просто обрезаем/дополняем
                    min_z = min(current_shape[0], target_shape[0])
                    min_y = min(current_shape[1], target_shape[1])
                    min_x = min(current_shape[2], target_shape[2])
                    result = np.zeros(target_shape, dtype=mask_data.dtype)
                    result[:min_z, :min_y, :min_x] = mask_data[:min_z, :min_y, :min_x]
                    self.log_message(f"3D маска обрезана до размеров изображения")
                    return result
                
                # Если обе маски 2D, но размеры не совпадают
                elif len(current_shape) == 2 and len(target_shape) == 2:
                    # Просто обрезаем/дополняем
                    min_y = min(current_shape[0], target_shape[0])
                    min_x = min(current_shape[1], target_shape[1])
                    result = np.zeros(target_shape, dtype=mask_data.dtype)
                    result[:min_y, :min_x] = mask_data[:min_y, :min_x]
                    self.log_message(f"2D маска обрезана до размеров изображения")
                    return result
            
            return mask_data
            
        except Exception as e:
            self.log_message(f"Ошибка при выравнивании размеров маски: {str(e)}")
            # В случае ошибки возвращаем маску с нулями нужного размера
            return np.zeros(target_shape, dtype=mask_data.dtype)
    
    def _clean_mask(self, mask_data: np.ndarray) -> np.ndarray:
        """
        Очистка маски от шумов с использованием морфологических операций.
        
        Args:
            mask_data: Исходная маска сегментации
            
        Returns:
            Очищенная маска
        """
        try:
            # Импортируем scipy для морфологических операций
            from scipy import ndimage
            
            # Удаляем маленькие объекты (шумы)
            try:
                # Для новых версий scipy
                labeled_mask, num_features = ndimage.label(mask_data, return_num=True)
            except (TypeError, ValueError):
                # Для старых версий scipy
                labeled_mask = ndimage.label(mask_data)
                # В старых версиях ndimage.label возвращает только массив меток
                # Количество объектов получаем как максимальное значение метки
                if hasattr(labeled_mask, 'max'):
                    num_features = labeled_mask.max()
                else:
                    # Если что-то пошло не так, считаем что объектов нет
                    num_features = 0
            
            if num_features > 0:
                # Вычисляем размеры каждого объекта
                object_sizes = ndimage.sum(mask_data, labeled_mask, range(1, num_features + 1))
                
                # Оставляем только объекты размером более 100 вокселей
                size_threshold = 100
                cleaned_mask = np.zeros_like(mask_data)
                
                for i, size in enumerate(object_sizes):
                    if size > size_threshold:
                        cleaned_mask[labeled_mask == (i + 1)] = 1
                
                # Применяем морфологическое закрытие для заполнения маленьких дыр
                cleaned_mask = ndimage.binary_closing(cleaned_mask, structure=np.ones((3, 3, 3)))
                
                # Применяем морфологическое открытие для удаления мелких выступов
                cleaned_mask = ndimage.binary_opening(cleaned_mask, structure=np.ones((3, 3, 3)))
                
                # Конвертируем обратно в uint8
                cleaned_mask = cleaned_mask.astype(np.uint8)
                
                self.log_message(f"Морфологическая очистка: удалено {num_features - len(object_sizes[object_sizes > size_threshold])} объектов")
                
                return cleaned_mask
            else:
                return mask_data
                
        except ImportError:
            self.log_message("Предупреждение: scipy недоступен, морфологическая очистка не выполнена")
            return mask_data
        except Exception as e:
            self.log_message(f"Ошибка при морфологической очистке: {str(e)}")
            return mask_data
    
    def _display_mask_as_volume(self, mask_data: np.ndarray):
        """
        Альтернативный метод отображения маски в виде объема.

        Args:
            mask_data: Маска сегментации
        """
        try:
            # Создание 3D объема для визуализации маски
            grid = pv.ImageData()
            grid.dimensions = np.array(mask_data.shape) + 1
            grid.spacing = (1.0, 1.0, 1.0)
            grid.origin = (0, 0, 0)
            grid.cell_data["mask"] = mask_data.ravel(order="F")

            # Отображение в виде объема с настройками для лучшей визуализации
            self.plotter.add_volume(
                grid,
                scalars="mask",
                cmap="Reds",
                opacity=self.mask_opacity,
                name="liver_mask_volume",
                clim=[0.5, 1.0],  # Отображаем только значения выше порога
                render=True
            )

            self.log_message("Маска отображена в виде объема (альтернативный метод)")

        except Exception as e:
            self.log_message(f"Ошибка при отображении маски в виде объема: {str(e)}")
            
    def calculate_liver_volume(self, mask_data: np.ndarray):
        """
        Расчет объема печени на основе маски сегментации.
        
        Args:
            mask_data: Маска сегментации
        """
        try:
            # Создание рабочего потока для расчета объема
            self.volume_calculation_worker = VolumeCalculationWorker()
            self.volume_calculation_worker.volume_calculated.connect(self.on_volume_calculated)
            self.volume_calculation_worker.error_occurred.connect(self.show_error)
            
            # Установка параметров и запуск потока
            self.volume_calculation_worker.calculate_volume(mask_data, self.current_voxel_volume)
            self.volume_calculation_worker.start()
            
        except Exception as e:
            self.show_error(f"Ошибка при расчете объема: {str(e)}")
            
    @pyqtSlot(float)
    def on_volume_calculated(self, volume_ml: float):
        """
        Обработчик завершения расчета объема печени.
        
        Args:
            volume_ml: Объем печени в миллилитрах
        """
        try:
            # Сохранение объема
            self.liver_volume_ml = volume_ml
            
            # Обновление метки
            self.volume_label.setText(f"Liver Volume: {volume_ml:.2f} ml")
            
            self.log_message(f"Объем печени рассчитан: {volume_ml:.2f} ml")
            
        except Exception as e:
            self.show_error(f"Ошибка при обработке рассчитанного объема: {str(e)}")
            
    @pyqtSlot(int)
    def update_progress(self, value: int):
        """
        Обновление прогресс-бара.
        
        Args:
            value: Значение прогресса (0-100)
        """
        self.progress_bar.setValue(value)
        
    @pyqtSlot(str)
    def log_message(self, message: str):
        """
        Добавление сообщения в лог.
        
        Args:
            message: Текст сообщения
        """
        self.logs_text.append(message)
        # Прокрутка к последнему сообщению
        scrollbar = self.logs_text.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())
        
    def show_error(self, error_message: str):
        """
        Отображение сообщения об ошибке.
        
        Args:
            error_message: Текст ошибки
        """
        # Логирование ошибки
        logger.error(error_message)
        self.log_message(f"ОШИБКА: {error_message}")
        
        # Отображение диалогового окна с ошибкой
        QMessageBox.critical(self, "Ошибка", error_message)
        
    def on_window_center_changed(self, value):
        """
        Обработчик изменения центра окна контраста.
        
        Args:
            value: Новое значение центра окна
        """
        self.window_center = value
        # Безопасная синхронизация слайдера и спинбокса без рекурсии
        self._sync_window_center_controls(value)
        
        # Отложенное обновление отображения для предотвращения зависаний
        self._schedule_display_update()
        
    def on_window_width_changed(self, value):
        """
        Обработчик изменения ширины окна контраста.
        
        Args:
            value: Новое значение ширины окна
        """
        self.window_width = value
        # Безопасная синхронизация слайдера и спинбокса без рекурсии
        self._sync_window_width_controls(value)
        
        # Отложенное обновление отображения для предотвращения зависаний
        self._schedule_display_update()
        
    def on_layer_visibility_changed(self):
        """Обработчик изменения видимости слоев."""
        self.show_ct_image = self.show_ct_checkbox.isChecked()
        self.show_mask = self.show_mask_checkbox.isChecked()
        
        # Отложенное обновление отображения для предотвращения зависаний
        self._schedule_display_update()
    
    def on_masked_region_visibility_changed(self, state):
        """Обработчик изменения видимости выделенной области."""
        self.show_masked_region = bool(state)
        self._schedule_display_update()
    
    def on_mask_display_mode_changed(self, mode_text):
        """Обработчик изменения режима отображения маски."""
        mode_map = {
            "Контур": "contour",
            "Полупрозрачное наложение": "overlay",
            "Только печень": "multiply"
        }
        self.mask_display_mode = mode_map.get(mode_text, "multiply")
        self._schedule_display_update()
        
    def on_mask_opacity_changed(self, value):
        """
        Обработчик изменения прозрачности маски.
        
        Args:
            value: Новое значение прозрачности (0.0-1.0)
        """
        self.mask_opacity = value
        # Безопасная синхронизация слайдера и спинбокса без рекурсии
        self._sync_mask_opacity_controls(value)
        
        # Обновление отображения маски (быстрая операция, можно выполнять синхронно)
        self.update_mask_opacity()
        
    def on_mask_opacity_slider_changed(self, value):
        """
        Обработчик изменения слайдера прозрачности маски.
        
        Args:
            value: Новое значение слайдера (0-100)
        """
        opacity = value / 100.0
        self.on_mask_opacity_changed(opacity)
        
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
        
    def update_display(self, model_type=None):
        """Обновление отображения с текущими настройками."""
        # Определяем, какие данные использовать - оригинальные или превью
        display_image_data = self.current_image_data
        display_mask_data = self.current_mask_data
        if self.preview_manager.use_preview_for_display:
            # Проверяем, есть ли доступное превью для текущего DICOM и разрешения
            if self.dicom_folder and self.current_series_id:
                preview_available = self.preview_manager.is_preview_available(
                    self.dicom_folder,
                    resolution=self.preview_manager.current_resolution,
                    series_id=self.current_series_id
                )
                
                if preview_available:
                    # Получаем превью из кэша
                    cache_key = f"{self.dicom_folder}_{self.preview_manager.current_resolution}_{self.current_series_id}"
                    cached_preview = self.preview_manager.preview_cache.get(cache_key)
                    if cached_preview and 'preview' in cached_preview:
                        display_image_data = cached_preview['preview']
                        
                        # Масштабируем маску под размер превью, если она доступна
                        if self.current_mask_data is not None:
                            from InferenceAnd3D.utils.mask_scaler import MaskScaler
                            mask_scaler = MaskScaler()
                            display_mask_data = mask_scaler.scale_mask(
                                self.current_mask_data,
                                display_image_data.shape
                            )
                        else:
                            display_mask_data = self.current_mask_data
         
        if display_image_data is not None:
            if self.view_mode == "2D":
                self._update_2d_display_with_data(display_image_data, display_mask_data)
            else:
                # Проверка размера изображения - для больших используем асинхронное обновление
                image_size = display_image_data.nbytes / (1024 * 1024)  # Размер в МБ
                
                if image_size > 50:  # Если изображение больше 50 МБ, используем асинхронное обновление
                    self._async_update_display_with_data(display_image_data, display_mask_data)
                else:
                    self._sync_update_display_with_data(display_image_data, display_mask_data, model_type)
                
    def _sync_update_display(self):
        """Синхронное обновление отображения для небольших изображений."""
        # Вызов метода с текущими данными
        self._sync_update_display_with_data(self.current_image_data, self.current_mask_data)
    
    def _sync_update_display_with_data(self, image_data, mask_data=None, model_type=None):
        """Синхронное обновление отображения для небольших изображений с заданными данными."""
        try:
            # Очистка сцены
            self.plotter.clear()
            
            # Отображение КТ изображения, если включено
            if self.show_ct_image and image_data is not None:
                self.display_volume(image_data)
            
            # Отображение маски, если включено и доступна
            if self.show_mask and mask_data is not None:
                # Используем сохраненный тип модели для цветовой дифференциации
                if model_type is None:
                    model_type = getattr(self, 'current_model_type', 'Unknown')
                self.display_mask(mask_data, model_type=model_type)
                
            # Отображение выделенной области, если включено
            if self.show_masked_region and image_data is not None and mask_data is not None:
                if model_type is None:
                    model_type = "Unknown"
                self.display_masked_region(image_data, mask_data, model_type)
                
                
                
            # Добавление координатных осей
            self.plotter.add_axes()
            
            # Установка камеры
            self.plotter.camera_position = 'iso'
            
            # Обновление сцены
            self.plotter.show()
        except Exception as e:
            self.show_error(f"Ошибка при обновлении отображения: {str(e)}")
    
        
            
    def display_masked_region(self, image_data, mask_data, model_type="Unknown"):
        """
        Отображение выделенной области печени на основе маски.
        
        Args:
            image_data: 3D массив данных изображения
            mask_data: 3D массив маски сегментации
            model_type: Тип модели для цветовой дифференциации
        """
        try:
            # Проверка на пустые данные
            if image_data is None or mask_data is None:
                self.log_message("Предупреждение: отсутствуют данные для отображения выделенной области")
                return
            
            # Убедимся, что маска бинарная (0 или 1)
            binary_mask = (mask_data > 0).astype(np.uint8)
            
            # Проверка размеров
            if image_data.shape != binary_mask.shape:
                self.log_message(f"Предупреждение: размеры изображения {image_data.shape} и маски {binary_mask.shape} не совпадают")
                return
            
            # Добавляем логирование для диагностики проблемы с режимом "multiply"
            self.log_message(f"Режим отображения: {self.mask_display_mode}")
            self.log_message(f"Форма image_data: {image_data.shape}, dtype: {image_data.dtype}")
            self.log_message(f"Форма mask_data: {mask_data.shape}, dtype: {mask_data.dtype}")
            self.log_message(f"Мин/макс image_data: {image_data.min()}/{image_data.max()}")
            self.log_message(f"Мин/макс mask_data: {mask_data.min()}/{mask_data.max()}")
            self.log_message(f"Количество ненулевых в маске: {np.count_nonzero(mask_data)}")
            
            # Применяем маску к изображению в зависимости от режима отображения
            if self.mask_display_mode == 'multiply':
                # Только область печени, остальное обнуляем
                masked_image = image_data * binary_mask
                # Добавляем логирование результата маскирования
                self.log_message(f"После маскирования - мин/макс masked_image: {masked_image.min()}/{masked_image.max()}")
                self.log_message(f"Количество ненулевых после маскирования: {np.count_nonzero(masked_image)}")
                
                # Создаем 3D объект для визуализации только ненулевых значений
                grid = pv.ImageData()
                grid.dimensions = np.array(masked_image.shape) + 1
                grid.spacing = (1.0, 1.0, 1.0)
                grid.origin = (0, 0, 0)
                
                # Используем только ненулевые значения для отображения
                # Если все значения нулевые, показываем предупреждение
                if np.count_nonzero(masked_image) == 0:
                    self.log_message("Предупреждение: все значения в masked_image равны 0")
                    return
                
                grid.cell_data["values"] = masked_image.ravel(order="F")
                
                # Используем параметры отображения, которые лучше подходят для режима "multiply"
                # Попробуем использовать surface mesh вместо volume rendering для лучшей видимости
                try:
                    # Используем contour для создания поверхностей только в областях с ненулевыми значениями
                    contour = grid.contour(isosurfaces=[0.5 * masked_image.max()])
                    if contour.n_points > 0:
                        # Отображаем как поверхность
                        self.plotter.add_mesh(
                            contour,
                            color="red",
                            opacity=self.mask_opacity,
                            name="masked_region",
                            show_scalar_bar=False
                        )
                    else:
                        # Если контур не создался, используем volume rendering
                        self.plotter.add_volume(
                            grid,
                            cmap="Reds",
                            opacity="sigmoid_6",  # Используем более высокий sigmoid для лучшей видимости
                            name="masked_region",
                            clim=[masked_image.min(), masked_image.max()]  # Устанавливаем правильный диапазон контраста
                        )
                except:
                    # Если контур не создался, используем volume rendering
                    self.plotter.add_volume(
                        grid,
                        cmap="Reds",
                        opacity="sigmoid_6",  # Используем более высокий sigmoid для лучшей видимости
                        name="masked_region",
                        clim=[masked_image.min(), masked_image.max()]  # Устанавливаем правильный диапазон контраста
                    )
                
            elif self.mask_display_mode == 'overlay':
                # Создаем полупрозрачное наложение
                from InferenceAnd3D.utils.mask_utils import create_transparent_overlay
                masked_image = image_data.copy()  # Создаем копию для безопасного изменения
                for i in range(masked_image.shape[0]):
                    slice_img = masked_image[i]
                    mask_slice = binary_mask[i]
                    # Применяем наложение к каждому срезу
                    masked_slice = create_transparent_overlay(slice_img, mask_slice, alpha=0.5)
                    masked_image[i] = masked_slice
                
                # Отображение в 3D
                grid = pv.ImageData()
                grid.dimensions = np.array(masked_image.shape) + 1
                grid.spacing = (1.0, 1.0, 1.0)
                grid.origin = (0, 0)
                grid.cell_data["values"] = masked_image.ravel(order="F")
                # Используем полупрозрачность для наложения
                opacity = self.mask_opacity if hasattr(self, 'mask_opacity') else 0.5
                self.plotter.add_volume(grid, cmap="Reds", opacity=opacity, name="masked_region")
                
            elif self.mask_display_mode == 'contour':
                # Отображаем только контур
                from InferenceAnd3D.utils.mask_utils import create_contour_overlay
                masked_image = image_data.copy()  # Создаем копию для безопасного изменения
                for i in range(masked_image.shape[0]):
                    slice_img = masked_image[i]
                    mask_slice = binary_mask[i]
                    # Создаем контурное наложение к каждому срезу
                    contour_slice = create_contour_overlay(slice_img, mask_slice)
                    masked_image[i] = contour_slice
                
                # Для контура используем специальное отображение
                grid = pv.ImageData()
                grid.dimensions = np.array(masked_image.shape) + 1
                grid.spacing = (1.0, 1.0, 1.0)
                grid.origin = (0, 0, 0)
                grid.cell_data["values"] = masked_image.ravel(order="F")
                # Отображаем как объем с высокой прозрачностью, чтобы видеть только контуры
                self.plotter.add_volume(grid, cmap="Reds", opacity="sigmoid_10", name="masked_region")
            else:
                # По умолчанию - умножение
                masked_image = image_data * binary_mask
                grid = pv.ImageData()
                grid.dimensions = np.array(masked_image.shape) + 1
                grid.spacing = (1.0, 1.0, 1.0)
                grid.origin = (0, 0, 0)
                grid.cell_data["values"] = masked_image.ravel(order="F")
                # Используем полупрозрачность для наложения
                opacity = self.mask_opacity if hasattr(self, 'mask_opacity') else 0.5
                self.plotter.add_volume(grid, cmap="Reds", opacity=opacity, name="masked_region")
                
        except Exception as e:
            self.show_error(f"Ошибка при отображении выделенной области: {str(e)}")
    
    def _async_update_display(self):
        """Асинхронное обновление отображения для больших изображений."""
        # Вызов метода с текущими данными
        self._async_update_display_with_data(self.current_image_data)

    def _async_update_display_with_data(self, image_data, mask_data=None):
        """Асинхронное обновление отображения для больших изображений с заданными данными."""
        try:
            # Остановка предыдущего рабочего потока, если он существует
            if self.display_update_worker and self.display_update_worker.isRunning():
                self.display_update_worker.stop()
                self.display_update_worker.wait()
                
            # Создание нового рабочего потока
            self.display_update_worker = DisplayUpdateWorker()
            self.display_update_worker.display_updated.connect(self._on_display_updated)
            self.display_update_worker.error_occurred.connect(self.show_error)
            
            # Установка параметров и запуск потока
            self.display_update_worker.set_parameters(
                image_data=image_data if image_data is not None else np.array([]),
                mask_data=mask_data,
                window_center=self.window_center,
                window_width=self.window_width,
                show_ct_image=self.show_ct_image,
                show_mask=self.show_mask,
                show_masked_region=getattr(self, 'show_masked_region', False),
                mask_display_mode=getattr(self, 'mask_display_mode', 'multiply'),
                mask_opacity=self.mask_opacity
            )
            self.display_update_worker.start()
            
            self.log_message("Запущено асинхронное обновление отображения...")
            
        except Exception as e:
            self.show_error(f"Ошибка при запуске асинхронного обновления: {str(e)}")
            
    @pyqtSlot(dict)
    def _on_display_updated(self, display_data: dict):
        """
        Обработчик завершения асинхронного обновления отображения.
        
        Args:
            display_data: Словарь с данными для отображения
        """
        try:
            # Очистка сцены
            self.plotter.clear()
            
            # Отображение КТ изображения, если включено
            if display_data['show_ct_image'] and display_data['image_data'] is not None:
                self.display_volume_optimized(display_data['image_data'])
            # Отображение маски, если включено и доступна
            if display_data['show_mask'] and display_data.get('mask_data') is not None:
                # Используем сохраненный тип модели для цветовой дифференциации
                model_type = getattr(self, 'current_model_type', 'Unknown')
                self.display_mask(display_data['mask_data'], model_type=model_type)
                
            # Отображение выделенной области, если включено
            if display_data.get('show_masked_region', False) and display_data['image_data'] is not None and display_data.get('mask_data') is not None:
                # Используем сохраненный тип модели для цветовой дифференциации
                model_type = getattr(self, 'current_model_type', 'Unknown')
                self.display_masked_region(display_data['image_data'], display_data['mask_data'], model_type)
                
            # Добавление координатных осей
            self.plotter.add_axes()
            
            # Установка камеры
            self.plotter.camera_position = 'iso'
            
            # Обновление сцены
            self.plotter.show()
            
            self.log_message("Отображение успешно обновлено")
            
        except Exception as e:
            self.show_error(f"Ошибка при обновлении отображения: {str(e)}")
            
    def update_mask_opacity(self):
        """Обновление прозрачности маски."""
        try:
            # Поиск актора маски на сцене
            if hasattr(self.plotter, 'renderer') and self.plotter.renderer:
                actors = self.plotter.renderer.actors
                for actor_name, actor in actors.items():
                    if actor_name == "liver_mask":
                        actor.GetProperty().SetOpacity(self.mask_opacity)
                        self.plotter.render()
                        break
        except Exception as e:
            logger.error(f"Ошибка при обновлении прозрачности маски: {str(e)}")
            
    def display_volume_optimized(self, volume_data: np.ndarray):
        """
        Оптимизированное отображение 3D объема в визуализаторе с прогрессивной загрузкой.
        
        Args:
            volume_data: 3D массив данных объема с уже примененными настройками контраста
        """
        try:
            # Проверка размера объема для определения стратегии отображения
            volume_size_mb = volume_data.nbytes / (1024 * 1024)
            
            if volume_size_mb > 100:  # Если объем больше 100 МБ, используем прогрессивную загрузку
                self._display_volume_progressive(volume_data)
            else:
                # Для небольших объемов используем стандартное отображение
                self._display_volume_standard(volume_data)
            
        except Exception as e:
            self.show_error(f"Ошибка при отображении объема: {str(e)}")
    
    def _display_volume_standard(self, volume_data: np.ndarray):
        """
        Стандартное отображение 3D объема.
        
        Args:
            volume_data: 3D массив данных объема
        """
        # Создание 3D объема для визуализации с оптимизацией
        grid = pv.ImageData()
        grid.dimensions = np.array(volume_data.shape) + 1
        grid.spacing = (1.0, 1.0, 1.0)  # Размер вокселя
        grid.origin = (0, 0, 0)
        
        # Оптимизированное присвоение данных - используем ravel() вместо flatten()
        grid.cell_data["values"] = volume_data.ravel(order="F")
        
        # Отображение объема с использованием volume rendering
        self.plotter.add_volume(grid, cmap="gray", opacity="sigmoid", name="volume")
    
    def _display_volume_progressive(self, volume_data: np.ndarray):
        """
        Прогрессивное отображение большого 3D объема.
        
        Args:
            volume_data: 3D массив данных объема
        """
        # Отключаем автоматическое уменьшение разрешения, чтобы использовать выбранное пользователем разрешение
        # Теперь сразу отображаем объем в полном разрешении
        self._display_volume_standard(volume_data)
    
    def _load_full_resolution(self, volume_data: np.ndarray):
        """
        Загрузка и отображение полного разрешения объема.
        
        Args:
            volume_data: 3D массив данных полного разрешения
        """
        try:
            # Очистка текущего отображения
            self.plotter.clear()
            
            # Отображение полного разрешения
            self._display_volume_standard(volume_data)
            
            self.log_message("Загружен объем полного разрешения")
        except Exception as e:
            self.show_error(f"Ошибка при загрузке полного разрешения: {str(e)}")
            
    def _sync_window_center_controls(self, value):
        """Безопасная синхронизация элементов управления центром окна без рекурсии."""
        # Блокировка сигналов для предотвращения рекурсии
        self.window_center_spin.blockSignals(True)
        self.window_center_slider.blockSignals(True)
        
        try:
            self.window_center_spin.setValue(value)
            self.window_center_slider.setValue(value)
        finally:
            # Восстановление сигналов
            self.window_center_spin.blockSignals(False)
            self.window_center_slider.blockSignals(False)
            
    def _sync_window_width_controls(self, value):
        """Безопасная синхронизация элементов управления шириной окна без рекурсии."""
        # Блокировка сигналов для предотвращения рекурсии
        self.window_width_spin.blockSignals(True)
        self.window_width_slider.blockSignals(True)
        
        try:
            self.window_width_spin.setValue(value)
            self.window_width_slider.setValue(value)
        finally:
            # Восстановление сигналов
            self.window_width_spin.blockSignals(False)
            self.window_width_slider.blockSignals(False)
            
    def _sync_mask_opacity_controls(self, value):
        """Безопасная синхронизация элементов управления прозрачностью маски без рекурсии."""
        # Блокировка сигналов для предотвращения рекурсии
        self.mask_opacity_spin.blockSignals(True)
        self.mask_opacity_slider.blockSignals(True)
        
        try:
            self.mask_opacity_spin.setValue(value)
            self.mask_opacity_slider.setValue(int(value * 100))
        finally:
            # Восстановление сигналов
            self.mask_opacity_spin.blockSignals(False)
            self.mask_opacity_slider.blockSignals(False)
            
    def on_view_mode_changed(self, mode_text):
        """Обработчик изменения режима просмотра."""
        if mode_text == "3D режим":
            self.view_mode = "3D"
            self.plotter.interactor.setVisible(True)
            self.graphics_view.setVisible(False)
            self.slice_navigation_group.setVisible(False)
        else:  # 2D режим
            self.view_mode = "2D"
            self.plotter.interactor.setVisible(False)
            self.graphics_view.setVisible(True)
            if self.current_image_data is not None and len(self.current_image_data.shape) == 3:
                self.slice_navigation_group.setVisible(True)
        
        self.update_display()
        self.log_message(f"Переключен на {mode_text}")
    
    def on_slice_changed(self, slice_value):
        """Обработчик изменения среза."""
        self.current_slice = slice_value
        self.update_slice_label()
        if self.view_mode == "2D":
            self._update_2d_display()
    
    def set_slice(self, slice_index):
        """Установка конкретного среза."""
        if 0 <= slice_index < self.max_slices:
            self.current_slice = slice_index
            self.slice_slider.setValue(slice_index)
            self.update_slice_label()
            if self.view_mode == "2D":
                self._update_2d_display()
    
    def update_slice_label(self):
        """Обновление метки с информацией о срезе."""
        self.slice_label.setText(f"{self.current_slice + 1} / {self.max_slices}")
    
    def _update_2d_display(self):
        """Обновление отображения в 2D режиме."""
        # Вызов метода с текущими данными
        self._update_2d_display_with_data(self.current_image_data, self.current_mask_data)

    def _update_2d_display_with_data(self, image_data, mask_data=None):
        """Обновление отображения в 2D режиме с заданными данными."""
        try:
            # Очистка сцены
            self.graphics_scene.clear()
            
            if image_data is None:
                return
            
            # Получение текущего среза
            if len(image_data.shape) == 3:
                slice_data = image_data[self.current_slice]
                mask_slice = mask_data[self.current_slice] if mask_data is not None else None
            else:
                slice_data = image_data
                mask_slice = mask_data
            
            # Применение настроек окна/уровня
            windowed_slice = self.apply_window_level(slice_data)
            
            # Нормализация в диапазон [0, 255] для отображения
            if windowed_slice.max() > windowed_slice.min():
                normalized_slice = ((windowed_slice - windowed_slice.min()) /
                                 (windowed_slice.max() - windowed_slice.min()) * 255).astype(np.uint8)
            else:
                normalized_slice = np.zeros_like(windowed_slice, dtype=np.uint8)
            
            # Создание изображения из numpy массива
            from PyQt6.QtGui import QImage, QPixmap, QColor
            from PyQt6.QtCore import Qt
            
            height, width = normalized_slice.shape
            bytes_per_line = width
            
            # Создание QImage в градациях серого
            q_image = QImage(normalized_slice.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            
            # Добавление маски, если она есть и включена
            if mask_slice is not None and self.show_mask:
                # Создание цветной маски
                mask_colored = np.zeros((height, width, 4), dtype=np.uint8)
                mask_colored[mask_slice > 0] = [255, 0, 0, int(self.mask_opacity * 255)]  # Красный с прозрачностью
                
                # Создание QImage для маски
                mask_q_image = QImage(mask_colored.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
                
                # Наложение маски на изображение
                from PyQt6.QtGui import QPainter
                painter = QPainter(q_image)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Overlay)
                painter.drawImage(0, 0, mask_q_image)
                painter.end()
            
            # Создание QPixmap и добавление на сцену
            pixmap = QPixmap.fromImage(q_image)
            self.graphics_scene.addPixmap(pixmap)
            
            # Настройка вида
            self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            
        except Exception as e:
            self.show_error(f"Ошибка при обновлении 2D отображения: {str(e)}")
    
    def _schedule_display_update(self):
        """Планирование отложенного обновления отображения (debounce)."""
        from PyQt6.QtCore import QTimer
        
        # Если уже есть отложенное обновление, отменяем его
        if self._update_timer is not None:
            self._update_timer.stop()
            
        # Создаем новый таймер для отложенного обновления
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update_display)
        self._update_timer.start(100)  # Задержка 100 мс для debounce
            
    def on_preview_resolution_changed(self, resolution_text):
        """Обработчик изменения разрешения превью."""
        # Преобразуем текст в значение, которое понимает PreviewManager
        resolution_map = {
            "Оригинал": "original",
            "128": 128,
            "256": 256,
            "512": 512
        }
        
        resolution = resolution_map.get(resolution_text, "original")
        self.preview_manager.set_resolution(resolution)
        self.log_message(f"Установлено разрешение превью: {resolution}")
        
        # Обновляем отображение, чтобы использовать новое разрешение
        if self.preview_manager.use_preview_for_display:
            self.update_display()
        
    def on_use_preview_changed(self, state):
        """Обработчик изменения состояния чекбокса использования превью."""
        use_preview = bool(state)
        self.preview_manager.update_use_preview_setting(use_preview)
        self.log_message(f"Использование превью для отображения: {use_preview}")
        
        # Обновляем отображение, чтобы отразить изменение настройки
        self.update_display()
        
    def on_update_preview_clicked(self):
        """Обработчик нажатия кнопки обновления превью."""
        if not self.dicom_folder or not self.current_series_id:
            self.show_error("Сначала загрузите DICOM серию")
            return
            
        try:
            # Обновляем превью в отдельном потоке
            self.log_message("Начата генерация превью...")
            
            # Получаем текущее разрешение
            current_resolution = self.preview_manager.current_resolution
            if current_resolution == 'original':
                resolution_text = 'original'
            else:
                resolution_text = str(current_resolution)
                
            # Генерируем превью
            # current_series_id - это идентификатор DICOM серии, а не путь к папке
            # Для генерации превью нужно использовать dicom_folder, а не пытаться создавать путь из серии
            dicom_series_path = os.path.normpath(self.dicom_folder)
            
            preview_data = self.preview_manager.generate_preview(dicom_series_path, resolution=current_resolution, series_id=self.current_series_id)
            
            self.log_message(f"Превью успешно сгенерировано для разрешения {resolution_text}")
            
            # Обновляем отображение, чтобы использовать новые данные превью
            self.update_display()
            
        except Exception as e:
            self.show_error(f"Ошибка при генерации превью: {str(e)}")
    
    def load_settings(self):
        """Загрузка настроек из QSettings."""
        try:
            self.window_center = self.settings.value("window_center", 0, type=int)
            self.window_width = self.settings.value("window_width", 400, type=int)
            self.show_ct_image = self.settings.value("show_ct_image", True, type=bool)
            self.show_mask = self.settings.value("show_mask", True, type=bool)
            self.mask_opacity = self.settings.value("mask_opacity", 0.5, type=float)
            
            # Загрузка настроек превью
            preview_resolution = self.settings.value("preview_resolution", "original", type=str)
            use_preview = self.settings.value("use_preview", True, type=bool)
            
            # Преобразуем сохраненное разрешение в подходящий формат
            # Если сохраненное значение - это число в виде строки, конвертируем в int
            if preview_resolution == "original":
                actual_resolution = "original"
            else:
                try:
                    actual_resolution = int(preview_resolution)
                except ValueError:
                    actual_resolution = "original"  # значение по умолчанию в случае ошибки
            
            # Применение настроек превью
            self.preview_manager.set_resolution(actual_resolution)
            self.preview_manager.update_use_preview_setting(use_preview)
            
            # Обновляем UI элементы
            resolution_map = {
                "original": "Оригинал",
                128: "128",
                256: "256",
                512: "512"
            }
            # Для поиска ключа в resolution_map используем actual_resolution
            preview_text = resolution_map.get(actual_resolution, "Оригинал")
            self.preview_resolution_combo.setCurrentText(preview_text)
            self.use_preview_checkbox.setChecked(use_preview)
            
            # Проверяем, инициализирован ли UI перед вызовом log_message
            if hasattr(self, 'logs_text'):
                self.log_message("Настройки успешно загружены")
        except Exception as e:
            logger.error(f"Ошибка при загрузке настроек: {str(e)}")
            
    def save_settings(self):
        """Сохранение настроек в QSettings."""
        try:
            self.settings.setValue("window_center", self.window_center)
            self.settings.setValue("window_width", self.window_width)
            self.settings.setValue("show_ct_image", self.show_ct_image)
            self.settings.setValue("show_mask", self.show_mask)
            self.settings.setValue("mask_opacity", self.mask_opacity)
            
            # Сохраняем настройки превью
            self.settings.setValue("preview_resolution", self.preview_manager.current_resolution)
            self.settings.setValue("use_preview", self.preview_manager.use_preview_for_display)
            
            # Проверяем, инициализирован ли UI перед вызовом log_message
            if hasattr(self, 'logs_text'):
                self.log_message("Настройки успешно сохранены")
        except Exception as e:
            logger.error(f"Ошибка при сохранении настроек: {str(e)}")
            
    def show_performance_report(self):
        """
        Отображение отчета о производительности.
        """
        try:
            # Генерация отчета о производительности
            report = performance_monitor.generate_report()
            
            # Отображение отчета в диалоговом окне
            QMessageBox.information(
                self,
                "Отчет о производительности",
                report
            )
            
            # Логирование отображения отчета
            self.log_message("Отчет о производительности отображен")
            
        except Exception as e:
            self.show_error(f"Ошибка при отображении отчета о производительности: {str(e)}")
            
    def clear_performance_metrics(self):
        """
        Очистка метрик производительности.
        """
        try:
            # Очистка всех метрик
            performance_monitor.clear_metrics()
            
            # Логирование очистки метрик
            self.log_message("Метрики производительности очищены")
            
        except Exception as e:
            self.show_error(f"Ошибка при очистке метрик производительности: {str(e)}")
            
    def closeEvent(self, event):
        """
        Обработчик закрытия окна приложения.
        
        Args:
            event: Событие закрытия
        """
        try:
            # Сохранение настроек
            self.save_settings()
            
            # Остановка всех рабочих потоков
            if self.dicom_loader_worker and self.dicom_loader_worker.isRunning():
                self.dicom_loader_worker.stop()
                self.dicom_loader_worker.wait()
                
            if self.segmentation_worker and self.segmentation_worker.isRunning():
                self.segmentation_worker.stop()
                self.segmentation_worker.wait()
                
            if self.volume_calculation_worker and self.volume_calculation_worker.isRunning():
                self.volume_calculation_worker.stop()
                self.volume_calculation_worker.wait()
                
            if self.display_update_worker and self.display_update_worker.isRunning():
                self.display_update_worker.stop()
                self.display_update_worker.wait()
                
            # Проверяем, инициализирован ли UI перед вызовом log_message
            if hasattr(self, 'logs_text'):
                self.log_message("Приложение закрывается...")
            
        except Exception as e:
            logger.error(f"Ошибка при закрытии приложения: {str(e)}")
            
        # Принятие события закрытия
        event.accept()


if __name__ == "__main__":
    # Создание приложения
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Создание и отображение главного окна
    window = MainWindow()
    window.show()
    
    # Запуск цикла обработки событий
    sys.exit(app.exec())