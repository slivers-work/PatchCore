import sys
import csv
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import shutil
import traceback
from pathlib import Path
from datetime import datetime


class DummyStream:
    def write(self, text):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False


if sys.stdout is None:
    sys.stdout = DummyStream()

if sys.stderr is None:
    sys.stderr = DummyStream()


def get_base_dir():
    """
    开发环境：返回当前 ui.py 所在目录
    打包后：返回 exe 所在目录
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = get_base_dir()

DEFAULT_INPUT_DIR = BASE_DIR / "input"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_CKPT_PATH = BASE_DIR / "ckpt" / "model.ckpt"


def ensure_default_dirs():
    DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "ckpt").mkdir(parents=True, exist_ok=True)


import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QFileDialog,
    QLabel,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QMessageBox,
    QTextEdit,
    QGroupBox,
    QFormLayout,
    QSplitter,
    QFrame,
)


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


# ============================================================
# 1. 模型加载线程：软件启动时加载一次 ckpt
# ============================================================

class ModelLoadWorker(QThread):
    log_signal = Signal(str)
    done_signal = Signal(object, object, str)
    error_signal = Signal(str)

    def __init__(self, ckpt_path):
        super().__init__()
        self.ckpt_path = ckpt_path

    def run(self):
        try:
            self.log_signal.emit("正在导入预测模块...")
            import torch
            import predict as core

            self.log_signal.emit("预测模块导入完成")
            self.log_signal.emit("正在加载 PatchCore 模型...")

            model = self.build_and_load_model(core, torch, self.ckpt_path)

            self.log_signal.emit("模型加载完成")
            self.done_signal.emit(core, model, self.ckpt_path)

        except Exception:
            self.error_signal.emit(traceback.format_exc())

    def build_and_load_model(self, core, torch, ckpt_path):
        """
        打包发布版本：
        不联网下载预训练权重，直接构建 pre_trained=False 的 PatchCore，
        然后从本地 model.ckpt 加载 state_dict。
        """
        from anomalib.models import Patchcore
        from anomalib.post_processing import PostProcessor

        post_processor = PostProcessor(
            image_sensitivity=core.IMAGE_SENSITIVITY,
            pixel_sensitivity=core.PIXEL_SENSITIVITY,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = Patchcore(
            backbone=core.BACKBONE,
            layers=core.LAYERS,
            pre_trained=False,
            coreset_sampling_ratio=core.CORESET_SAMPLING_RATIO,
            num_neighbors=core.NUM_NEIGHBORS,
            post_processor=post_processor,
        )

        ckpt = torch.load(
            str(ckpt_path),
            map_location="cpu",
            weights_only=False,
        )

        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device)
        return model


# ============================================================
# 2. 预测线程：逐张预测，预测完一张立即把结果发给 UI
# ============================================================

class PredictWorker(QThread):
    log_signal = Signal(str)
    one_result_signal = Signal(dict)
    done_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(
        self,
        core,
        model,
        raw_input_path,
        ckpt_path,
        output_dir,
        custom_threshold,
        save_processed_images,
        clear_processed_dir,
        save_patchcore_vis,
        clear_patchcore_result_dir,
        use_stamped_patchcore_dir,
    ):
        super().__init__()
        self.core = core
        self.model = model
        self.raw_input_path = raw_input_path
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.custom_threshold = custom_threshold
        self.save_processed_images = save_processed_images
        self.clear_processed_dir = clear_processed_dir
        self.save_patchcore_vis = save_patchcore_vis
        self.clear_patchcore_result_dir = clear_patchcore_result_dir
        self.use_stamped_patchcore_dir = use_stamped_patchcore_dir

    def run(self):
        try:
            csv_path = self.run_predict_with_cached_model()
            self.done_signal.emit(csv_path)
        except Exception:
            self.error_signal.emit(traceback.format_exc())

    def run_predict_with_cached_model(self):
        from anomalib.data import PredictDataset
        import torch

        core = self.core
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ====================================================
        # 设置 predict.py 的全局参数
        # ====================================================
        core.STAMP = stamp
        core.RAW_INPUT_PATH = Path(self.raw_input_path)
        core.CKPT_PATH = str(self.ckpt_path)

        # 兼容 predict.py 里原来的变量名：SAVE_CROPPED_IMAGES 实际表示是否保留预处理图
        core.SAVE_CROPPED_IMAGES = bool(self.save_processed_images)
        core.CLEAR_CROPPED_DIR = bool(self.clear_processed_dir)

        core.CROPPED_OUTPUT_DIR = output_dir / "processed_full_5472x3648"
        core.TEMP_CROPPED_OUTPUT_DIR = output_dir / "_temp_full_predict"

        core.OUT_CSV_BASE = output_dir / "predict.csv"
        core.OUT_CSV = core.OUT_CSV_BASE.with_name(
            f"{core.OUT_CSV_BASE.stem}_{stamp}{core.OUT_CSV_BASE.suffix}"
        )
        core.CUSTOM_SCORE_THRESHOLD = self.custom_threshold

        # ====================================================
        # 热力图输出控制
        # ====================================================
        core.PATCHCORE_RESULT_ROOT = output_dir / "patchcore_vis"
        if self.save_patchcore_vis:
            if self.clear_patchcore_result_dir and core.PATCHCORE_RESULT_ROOT.exists():
                shutil.rmtree(core.PATCHCORE_RESULT_ROOT)

            if self.use_stamped_patchcore_dir:
                core.PATCHCORE_RUN_ROOT = core.PATCHCORE_RESULT_ROOT / f"run_{stamp}"
            else:
                core.PATCHCORE_RUN_ROOT = core.PATCHCORE_RESULT_ROOT

            core.PATCHCORE_RUN_ROOT.mkdir(parents=True, exist_ok=True)
        else:
            core.PATCHCORE_RUN_ROOT = output_dir / "_temp_patchcore_vis"
            if core.PATCHCORE_RUN_ROOT.exists():
                shutil.rmtree(core.PATCHCORE_RUN_ROOT)
            core.PATCHCORE_RUN_ROOT.mkdir(parents=True, exist_ok=True)

        # ====================================================
        # 1. 图片尺寸检查 / 预处理
        # ====================================================
        self.log_signal.emit("开始图片尺寸检查 / 预处理...")
        processed_paths, predict_image_dir = core.prepare_cropped_images()

        if not processed_paths:
            raise RuntimeError("没有可预测的图片，请检查输入路径。")

        self.log_signal.emit(f"预处理完成：{len(processed_paths)} 张")
        self.log_signal.emit("开始 PatchCore 推理，当前使用已加载模型，不重复加载 ckpt。")

        # ====================================================
        # 2. 逐张预测：第一张完成后马上通知 UI，不等全部完成
        # ====================================================
        engine = self.create_engine(core, save_vis=self.save_patchcore_vis)
        rows = []

        for idx, img_path in enumerate(processed_paths):
            dataset = PredictDataset(
                path=Path(img_path),
                image_size=core.IMAGE_SIZE,
            )

            with torch.no_grad():
                predictions = engine.predict(
                    model=self.model,
                    dataset=dataset,
                    ckpt_path=None,
                    return_predictions=True,
                )

            if predictions is None:
                self.log_signal.emit(f"[WARN] 未拿到预测结果：{Path(img_path).name}")
                continue

            one_rows = self.parse_predictions(predictions, core)
            for row in one_rows:
                rows.append(row)
                self.one_result_signal.emit(row)

            self.log_signal.emit(f"进度：{idx + 1}/{len(processed_paths)}")

        # ====================================================
        # 3. 保存 CSV
        # ====================================================
        self.save_csv(core.OUT_CSV, rows)

        # ====================================================
        # 4. 删除临时目录
        # ====================================================
        if not self.save_processed_images:
            try:
                if Path(predict_image_dir).exists():
                    shutil.rmtree(predict_image_dir)
                    self.log_signal.emit("已删除临时预处理目录")
            except Exception as e:
                self.log_signal.emit(f"[WARN] 删除临时预处理目录失败: {e}")

        if not self.save_patchcore_vis:
            try:
                if Path(core.PATCHCORE_RUN_ROOT).exists():
                    shutil.rmtree(core.PATCHCORE_RUN_ROOT)
                    self.log_signal.emit("已删除临时热力图目录")
            except Exception as e:
                self.log_signal.emit(f"[WARN] 删除临时热力图目录失败: {e}")

        self.log_signal.emit(f"预测完成，CSV：{core.OUT_CSV}")
        return str(core.OUT_CSV)

    def create_engine(self, core, save_vis):
        from anomalib.engine import Engine

        accelerator = "gpu"
        devices = 1

        try:
            import torch
            if not torch.cuda.is_available():
                accelerator = "cpu"
                devices = 1
        except Exception:
            accelerator = "cpu"
            devices = 1

        if not save_vis:
            try:
                return Engine(
                    default_root_dir=str(core.PATCHCORE_RUN_ROOT),
                    accelerator=accelerator,
                    devices=devices,
                    callbacks=[],
                    enable_progress_bar=False,
                    logger=False,
                )
            except TypeError:
                return Engine(
                    default_root_dir=str(core.PATCHCORE_RUN_ROOT),
                    accelerator=accelerator,
                    devices=devices,
                )

        try:
            return Engine(
                default_root_dir=str(core.PATCHCORE_RUN_ROOT),
                accelerator=accelerator,
                devices=devices,
                enable_progress_bar=False,
                logger=False,
            )
        except TypeError:
            return Engine(
                default_root_dir=str(core.PATCHCORE_RUN_ROOT),
                accelerator=accelerator,
                devices=devices,
            )

    def parse_predictions(self, predictions, core):
        rows = []

        for batch in predictions:
            image_paths = self.path_to_list(self.get_batch_attr(batch, "image_path"))
            pred_labels = self.value_to_list(self.get_batch_attr(batch, "pred_label"))
            pred_scores = self.value_to_list(self.get_batch_attr(batch, "pred_score"))

            for idx, img_path in enumerate(image_paths):
                label_int = int(pred_labels[idx])
                score_float = float(pred_scores[idx])

                if core.CUSTOM_SCORE_THRESHOLD is None:
                    final_result = "NG" if label_int == 1 else "OK"
                else:
                    final_result = "NG" if score_float >= core.CUSTOM_SCORE_THRESHOLD else "OK"

                filename = Path(img_path).name

                rows.append({
                    "filename": filename,
                    "image_path": str(img_path),
                    "final_result": final_result,
                    "pred_label": label_int,
                    "pred_score": score_float,
                })

        return rows

    def save_csv(self, csv_path, rows):
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "filename",
            "image_path",
            "final_result",
            "pred_label",
            "pred_score",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def get_batch_attr(batch, *names):
        for name in names:
            if hasattr(batch, name):
                return getattr(batch, name)
            if isinstance(batch, dict) and name in batch:
                return batch[name]
        return None

    @staticmethod
    def value_to_list(x):
        if x is None:
            return []
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        if hasattr(x, "tolist"):
            x = x.tolist()
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    @staticmethod
    def path_to_list(x):
        if isinstance(x, (list, tuple)):
            return [str(p) for p in x]
        return [str(x)]


# ============================================================
# 3. 主窗口
# ============================================================

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("OK/NG检测识别")
        self.resize(1500, 900)

        self.core = None
        self.model = None
        self.loaded_ckpt_path = None

        self.model_load_worker = None
        self.predict_worker = None

        self.current_images = []
        self.current_index = -1
        self.last_result_map = {}

        self.live_total = 0
        self.live_done = 0
        self.live_ok_count = 0
        self.live_ng_count = 0

        ensure_default_dirs()

        self.input_edit = QLineEdit(str(DEFAULT_INPUT_DIR))
        self.ckpt_edit = QLineEdit(str(DEFAULT_CKPT_PATH))
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        self.threshold_edit = QLineEdit("0.358")

        self.save_processed_cb = QCheckBox("保留预处理后的完整图片")
        self.save_processed_cb.setChecked(False)

        self.clear_processed_cb = QCheckBox("运行前清空预处理图目录")
        self.clear_processed_cb.setChecked(True)

        self.save_heatmap_cb = QCheckBox("保存 PatchCore 热力图/结果图")
        self.save_heatmap_cb.setChecked(False)

        self.clear_heatmap_cb = QCheckBox("运行前清空热力图目录")
        self.clear_heatmap_cb.setChecked(True)

        self.stamped_heatmap_cb = QCheckBox("热力图按时间新建 run 文件夹")
        self.stamped_heatmap_cb.setChecked(True)

        self.init_ui()
        self.apply_dark_style()
        self.start_load_model()

    # ========================================================
    # UI 布局
    # ========================================================

    def init_ui(self):
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        preview_group = QGroupBox("图片预览")
        preview_layout = QVBoxLayout(preview_group)

        preview_center_layout = QHBoxLayout()

        self.prev_btn = QPushButton("上一张")
        self.prev_btn.setFixedSize(110, 46)
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.prev_btn.setEnabled(False)

        self.image_label = QLabel("请先选择输入图片或输入文件夹")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(850, 560)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setStyleSheet("font-size: 18px; color: #d0d0d0;")

        self.next_btn = QPushButton("下一张")
        self.next_btn.setFixedSize(110, 46)
        self.next_btn.clicked.connect(self.show_next_image)
        self.next_btn.setEnabled(False)

        preview_center_layout.addWidget(self.prev_btn)
        preview_center_layout.addWidget(self.image_label, stretch=1)
        preview_center_layout.addWidget(self.next_btn)

        self.index_label = QLabel("0 / 0")
        self.index_label.setAlignment(Qt.AlignCenter)
        self.index_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        preview_layout.addLayout(preview_center_layout, stretch=1)
        preview_layout.addWidget(self.index_label)

        left_layout.addWidget(preview_group, stretch=4)

        bottom_split = QSplitter(Qt.Horizontal)

        result_group = QGroupBox("输出结果")
        result_layout = QVBoxLayout(result_group)

        self.result_label = QLabel("None")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 64px; font-weight: bold; color: #999999;")
        result_layout.addWidget(self.result_label)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)

        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.clear_log)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        log_layout.addWidget(self.clear_log_btn, alignment=Qt.AlignRight)
        log_layout.addWidget(self.log_box)

        bottom_split.addWidget(result_group)
        bottom_split.addWidget(log_group)
        bottom_split.setStretchFactor(0, 1)
        bottom_split.setStretchFactor(1, 1)

        left_layout.addWidget(bottom_split, stretch=2)

        right_panel = QWidget()
        right_panel.setFixedWidth(360)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        path_group = QGroupBox("路径配置")
        form = QFormLayout(path_group)

        form.addRow("输入路径：", self.input_edit)

        self.choose_file_btn = QPushButton("选择单张图片")
        self.choose_file_btn.clicked.connect(self.choose_file)
        form.addRow("", self.choose_file_btn)

        self.choose_folder_btn = QPushButton("选择输入文件夹")
        self.choose_folder_btn.clicked.connect(self.choose_folder)
        form.addRow("", self.choose_folder_btn)

        form.addRow("模型 ckpt：", self.ckpt_edit)

        self.choose_ckpt_btn = QPushButton("选择模型")
        self.choose_ckpt_btn.clicked.connect(self.choose_ckpt)
        form.addRow("", self.choose_ckpt_btn)

        self.reload_model_btn = QPushButton("重新加载模型")
        self.reload_model_btn.clicked.connect(self.start_load_model)
        form.addRow("", self.reload_model_btn)

        form.addRow("输出文件夹：", self.output_edit)

        self.choose_output_btn = QPushButton("选择输出文件夹")
        self.choose_output_btn.clicked.connect(self.choose_output_dir)
        form.addRow("", self.choose_output_btn)

        form.addRow("自定义阈值：", self.threshold_edit)
        right_layout.addWidget(path_group)

        output_group = QGroupBox("输出控制")
        output_layout = QVBoxLayout(output_group)
        output_layout.addWidget(self.save_processed_cb)
        output_layout.addWidget(self.clear_processed_cb)
        output_layout.addWidget(self.save_heatmap_cb)
        output_layout.addWidget(self.clear_heatmap_cb)
        output_layout.addWidget(self.stamped_heatmap_cb)
        right_layout.addWidget(output_group)

        self.single_btn = QPushButton("单张识别")
        self.single_btn.setFixedHeight(52)
        self.single_btn.clicked.connect(self.start_single_predict)
        self.single_btn.setEnabled(False)

        self.batch_btn = QPushButton("批量识别")
        self.batch_btn.setFixedHeight(52)
        self.batch_btn.clicked.connect(self.start_batch_predict)
        self.batch_btn.setEnabled(False)

        self.open_output_btn = QPushButton("打开输出目录")
        self.open_output_btn.setFixedHeight(42)
        self.open_output_btn.clicked.connect(self.open_output_dir)

        right_layout.addWidget(self.single_btn)
        right_layout.addWidget(self.batch_btn)
        right_layout.addWidget(self.open_output_btn)

        self.state_label = QLabel("State: loading model")
        self.state_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffaa00;")
        right_layout.addWidget(self.state_label)
        right_layout.addStretch()

        root_layout.addWidget(left_panel, stretch=1)
        root_layout.addWidget(right_panel)

    def apply_dark_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1f1f1f;
                color: #eeeeee;
                font-family: Microsoft YaHei;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #666666;
                margin-top: 10px;
                padding: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLineEdit, QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
            }
            QPushButton {
                background-color: #2f7de1;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #3a3a3a;
                color: #888888;
            }
            QPushButton:hover:enabled {
                background-color: #4090ff;
            }
            QCheckBox {
                spacing: 8px;
            }
        """)

    # ========================================================
    # 图片选择与预览
    # ========================================================

    def collect_images(self, path):
        path = Path(path)
        if path.is_file():
            return [path] if path.suffix.lower() in IMAGE_EXTS else []

        images = []
        if path.is_dir():
            for ext in IMAGE_EXTS:
                images.extend(path.glob(f"*{ext}"))
                images.extend(path.glob(f"*{ext.upper()}"))
        return sorted(list(set(images)))

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if path:
            self.input_edit.setText(path)
            self.load_images_for_preview(path)

    def choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self.input_edit.setText(path)
            self.load_images_for_preview(path)

    def load_images_for_preview(self, path):
        self.current_images = self.collect_images(path)
        self.last_result_map = {}
        if not self.current_images:
            self.current_index = -1
            self.image_label.setText("没有找到图片")
            self.index_label.setText("0 / 0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.set_result_none("None")
            return

        self.current_index = 0
        self.show_current_image()
        self.update_nav_buttons()

    def show_current_image(self):
        if self.current_index < 0 or self.current_index >= len(self.current_images):
            return

        img_path = self.current_images[self.current_index]
        pixmap = QPixmap(str(img_path))

        if pixmap.isNull():
            self.image_label.setText(f"图片读取失败：{img_path.name}")
            return

        scaled = pixmap.scaled(
            self.image_label.width() - 20,
            self.image_label.height() - 20,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.index_label.setText(f"{self.current_index + 1} / {len(self.current_images)}")
        self.update_result_for_current_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_images:
            self.show_current_image()

    def show_prev_image(self):
        if not self.current_images:
            return
        self.current_index = max(0, self.current_index - 1)
        self.show_current_image()
        self.update_nav_buttons()

    def show_next_image(self):
        if not self.current_images:
            return
        self.current_index = min(len(self.current_images) - 1, self.current_index + 1)
        self.show_current_image()
        self.update_nav_buttons()

    def update_nav_buttons(self):
        total = len(self.current_images)
        self.prev_btn.setEnabled(total > 1 and self.current_index > 0)
        self.next_btn.setEnabled(total > 1 and self.current_index < total - 1)

    # ========================================================
    # 路径/模型
    # ========================================================

    def choose_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 model.ckpt",
            "",
            "Checkpoint (*.ckpt);;All Files (*.*)",
        )
        if path:
            self.ckpt_edit.setText(path)
            self.start_load_model()

    def choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_edit.setText(path)

    def open_output_dir(self):
        output_dir = Path(self.output_edit.text())
        output_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(output_dir))

    def start_load_model(self):
        ckpt_path = self.ckpt_edit.text().strip()
        if not ckpt_path:
            QMessageBox.warning(self, "提示", "请先选择 model.ckpt")
            return
        if not Path(ckpt_path).exists():
            QMessageBox.warning(self, "提示", f"模型文件不存在：\n{ckpt_path}")
            return

        self.single_btn.setEnabled(False)
        self.batch_btn.setEnabled(False)
        self.state_label.setText("State: loading model")
        self.state_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffaa00;")
        self.set_result_none("Loading")
        self.add_log("开始加载模型...")

        self.core = None
        self.model = None
        self.loaded_ckpt_path = None

        self.model_load_worker = ModelLoadWorker(ckpt_path)
        self.model_load_worker.log_signal.connect(self.add_log)
        self.model_load_worker.done_signal.connect(self.on_model_loaded)
        self.model_load_worker.error_signal.connect(self.on_model_load_error)
        self.model_load_worker.start()

    def on_model_loaded(self, core, model, ckpt_path):
        self.core = core
        self.model = model
        self.loaded_ckpt_path = ckpt_path

        # 读取 predict.py 里的默认 CUSTOM_SCORE_THRESHOLD，并显示到 UI
        default_threshold = getattr(core, "CUSTOM_SCORE_THRESHOLD", None)
        if default_threshold is None:
            self.threshold_edit.setText("None")
        else:
            self.threshold_edit.setText(str(default_threshold))

        self.single_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self.state_label.setText("State: model ready")
        self.state_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #00aa00;")
        self.set_result_none("None")
        self.add_log("模型初始化完成")

    def on_model_load_error(self, error_text):
        self.core = None
        self.model = None
        self.loaded_ckpt_path = None
        self.single_btn.setEnabled(False)
        self.batch_btn.setEnabled(False)
        self.state_label.setText("State: model failed")
        self.state_label.setStyleSheet("font-size: 20px; font-weight: bold; color: red;")
        self.set_result_ng("模型加载失败")
        self.log_box.append(error_text)
        QMessageBox.critical(self, "模型加载失败", error_text)

    # ========================================================
    # 识别流程
    # ========================================================

    def parse_threshold(self):
        text = self.threshold_edit.text().strip()
        if text == "" or text.lower() == "none":
            return None
        return float(text)

    def start_single_predict(self):
        path = self.input_edit.text().strip()
        if Path(path).is_dir() and self.current_images and self.current_index >= 0:
            path = str(self.current_images[self.current_index])
        self.start_predict(path)

    def start_batch_predict(self):
        path = self.input_edit.text().strip()
        self.start_predict(path)

    def start_predict(self, raw_input_path):
        if self.core is None or self.model is None:
            QMessageBox.warning(self, "提示", "模型还没有加载完成")
            return

        raw_input_path = str(raw_input_path).strip()
        ckpt_path = self.ckpt_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        if not raw_input_path or not Path(raw_input_path).exists():
            QMessageBox.warning(self, "提示", f"输入路径不存在：\n{raw_input_path}")
            return

        if ckpt_path != self.loaded_ckpt_path:
            QMessageBox.warning(self, "提示", "当前 ckpt 路径和已加载模型不一致，请重新加载模型。")
            return

        if not output_dir:
            QMessageBox.warning(self, "提示", "请选择输出目录")
            return

        try:
            custom_threshold = self.parse_threshold()
        except Exception:
            QMessageBox.warning(self, "提示", "自定义阈值只能填写 None 或数字，例如 0.43")
            return

        self.single_btn.setEnabled(False)
        self.batch_btn.setEnabled(False)
        self.state_label.setText("State: running")
        self.state_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffaa00;")
        self.set_result_none("检测中")
        self.add_log("开始检测...")

        self.last_result_map = {}
        self.live_total = len(self.collect_images(raw_input_path))
        self.live_done = 0
        self.live_ok_count = 0
        self.live_ng_count = 0

        self.predict_worker = PredictWorker(
            core=self.core,
            model=self.model,
            raw_input_path=raw_input_path,
            ckpt_path=ckpt_path,
            output_dir=output_dir,
            custom_threshold=custom_threshold,
            save_processed_images=self.save_processed_cb.isChecked(),
            clear_processed_dir=self.clear_processed_cb.isChecked(),
            save_patchcore_vis=self.save_heatmap_cb.isChecked(),
            clear_patchcore_result_dir=self.clear_heatmap_cb.isChecked(),
            use_stamped_patchcore_dir=self.stamped_heatmap_cb.isChecked(),
        )
        self.predict_worker.log_signal.connect(self.add_log)
        self.predict_worker.one_result_signal.connect(self.on_one_result)
        self.predict_worker.done_signal.connect(self.on_predict_done)
        self.predict_worker.error_signal.connect(self.on_predict_error)
        self.predict_worker.start()

    def on_one_result(self, row):
        filename = Path(row.get("filename", "")).name
        result = row.get("final_result", "")

        if filename:
            self.last_result_map[filename] = result

        self.live_done += 1
        if result == "OK":
            self.live_ok_count += 1
        elif result == "NG":
            self.live_ng_count += 1

        self.add_log(
            f"单张完成：{filename} -> {result}，"
            f"进度 {self.live_done}/{self.live_total}，"
            f"OK={self.live_ok_count}，NG={self.live_ng_count}"
        )

        self.update_result_for_current_image()

    def on_predict_done(self, csv_path):
        self.single_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self.state_label.setText("State: model ready")
        self.state_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #00aa00;")
        self.add_log(f"检测完成，CSV：{csv_path}")
        self.add_log(f"批量统计：OK={self.live_ok_count}，NG={self.live_ng_count}，总数={self.live_done}")
        self.update_result_for_current_image()

    def on_predict_error(self, error_text):
        self.single_btn.setEnabled(True)
        self.batch_btn.setEnabled(True)
        self.state_label.setText("State: error")
        self.state_label.setStyleSheet("font-size: 20px; font-weight: bold; color: red;")
        self.set_result_ng("检测失败")
        self.log_box.append(error_text)
        QMessageBox.critical(self, "检测失败", error_text)

    # ========================================================
    # 结果显示
    # ========================================================

    def update_result_for_current_image(self):
        if not self.current_images or self.current_index < 0:
            return

        current_path = self.current_images[self.current_index]
        current_name = current_path.name
        current_stem = current_path.stem

        result = None

        if current_name in self.last_result_map:
            result = self.last_result_map[current_name]

        if result is None:
            for name, value in self.last_result_map.items():
                if Path(name).stem == current_stem:
                    result = value
                    break

        if result == "OK":
            self.set_result_ok("OK")
        elif result == "NG":
            self.set_result_ng("NG")
        else:
            self.set_result_none("None")

    def set_result_ok(self, text="OK"):
        self.result_label.setText(text)
        self.result_label.setStyleSheet("font-size: 76px; font-weight: bold; color: #00cc44;")

    def set_result_ng(self, text="NG"):
        self.result_label.setText(text)
        self.result_label.setStyleSheet("font-size: 76px; font-weight: bold; color: red;")

    def set_result_none(self, text="None"):
        self.result_label.setText(text)
        self.result_label.setStyleSheet("font-size: 64px; font-weight: bold; color: #999999;")

    # ========================================================
    # 日志
    # ========================================================

    def add_log(self, text):
        now = datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{now}] {text}")

    def clear_log(self):
        self.log_box.clear()


# ============================================================
# 4. 程序入口
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
