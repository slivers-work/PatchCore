from pathlib import Path
import sys
import os
import csv
import shutil
from datetime import datetime

import cv2
import numpy as np
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.post_processing import PostProcessor


# ============================================================
# 0. 离线与打包路径工具函数
# ============================================================
# 预测端不需要联网下载预训练权重，防止客户电脑访问 HuggingFace 超时
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def get_base_dir():
    """
    开发环境：返回当前 py 文件所在目录
    PyInstaller 打包后：返回 exe 所在目录
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = get_base_dir()


# ============================================================
# 1. 输入输出路径
# ============================================================
RAW_INPUT_PATH = BASE_DIR / "input"

# 时间戳：CSV / 热力图结果按时间区分
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# 1.1 预处理图输出控制
# ============================================================
# 这里不再裁剪，只做尺寸检查和 resize。
# 为了兼容原来的 UI，变量名仍然保留 SAVE_CROPPED_IMAGES / CROPPED_OUTPUT_DIR。
SAVE_CROPPED_IMAGES = False

# 如果 SAVE_CROPPED_IMAGES = True，就保存 resize 后的完整原图到这里
CROPPED_OUTPUT_DIR = BASE_DIR / "output" / "processed_full_5472x3648"

# 如果 SAVE_CROPPED_IMAGES = False，就用这个临时目录，预测完成后自动删除
TEMP_CROPPED_OUTPUT_DIR = BASE_DIR / "output" / "_temp_full_predict"

# 是否每次运行前清空预处理输出目录
CLEAR_CROPPED_DIR = True


# ============================================================
# 1.2 CSV 输出控制
# ============================================================
OUT_CSV_BASE = BASE_DIR / "output" / "predict.csv"
OUT_CSV = OUT_CSV_BASE.with_name(
    f"{OUT_CSV_BASE.stem}_{STAMP}{OUT_CSV_BASE.suffix}"
)


# ============================================================
# 1.3 PatchCore 热力图 / 结果图输出控制
# ============================================================
PATCHCORE_RESULT_ROOT = BASE_DIR / "output" / "patchcore_vis"
CLEAR_PATCHCORE_RESULT_DIR = True
USE_STAMPED_PATCHCORE_DIR = False

if USE_STAMPED_PATCHCORE_DIR:
    PATCHCORE_RUN_ROOT = PATCHCORE_RESULT_ROOT / f"run_{STAMP}"
else:
    PATCHCORE_RUN_ROOT = PATCHCORE_RESULT_ROOT

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


# ============================================================
# 2. PatchCore 模型路径
# ============================================================
# 默认模型路径：exe 同级目录下 ckpt/model.ckpt
CKPT_PATH = str(BASE_DIR / "ckpt" / "model.ckpt")


# ============================================================
# 3. 原图尺寸配置
# ============================================================
# OpenCV 读取后：img.shape = (height, width, channel)
# 你说的图片尺寸 (5472, 3648) 按常见写法理解为：width=5472, height=3648
TARGET_WIDTH = 5472
TARGET_HEIGHT = 3648

# OpenCV resize 使用 (width, height)
TARGET_SIZE_WH = (TARGET_WIDTH, TARGET_HEIGHT)

# Anomalib / torch transform 通常使用 (height, width)
IMAGE_SIZE = (TARGET_HEIGHT, TARGET_WIDTH)


# ============================================================
# 4. PatchCore 参数：必须和训练时一致
# ============================================================
IMAGE_SENSITIVITY = 0.60
PIXEL_SENSITIVITY = 0.70

BACKBONE = "resnet18"
LAYERS = ("layer2", "layer3")
CORESET_SAMPLING_RATIO = 0.005
NUM_NEIGHBORS = 1


# ============================================================
# 5. 一级分类阈值
# ============================================================
# None：使用 Anomalib 默认 pred_label
# 数字：使用 pred_score 自己判断
CUSTOM_SCORE_THRESHOLD = 0.358


# ============================================================
# 6. 二级规则开关
# ============================================================
# 这版先去掉二级规则，只保留一级分类。
# 保留变量是为了兼容 UI 代码里的 hasattr(core, "ENABLE_STAGE2_RULE") 判断。
ENABLE_STAGE2_RULE = False


# ============================================================
# 7. 工具函数：收集图片
# ============================================================
def collect_images(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTS:
            return [input_path]
        return []

    images = []
    if input_path.is_dir():
        for ext in IMAGE_EXTS:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))

    return sorted(list(set(images)))


# ============================================================
# 8. 预处理：不裁剪，只检查原图尺寸并 resize 到 5472x3648
# ============================================================
def resize_full_image_if_needed(img, img_path=None):
    """
    输入任意尺寸图片。
    如果不是 width=5472, height=3648，就 resize 到 5472x3648。
    不做裁剪，不做圆片 mask，不屏蔽镭刻码。
    """
    h, w = img.shape[:2]

    if (w, h) == TARGET_SIZE_WH:
        return img.copy(), False

    resized = cv2.resize(
        img,
        TARGET_SIZE_WH,
        interpolation=cv2.INTER_AREA,
    )

    if img_path is not None:
        print(
            f"[RESIZE] {Path(img_path).name}: "
            f"原尺寸(width={w}, height={h}) -> "
            f"目标尺寸(width={TARGET_WIDTH}, height={TARGET_HEIGHT})"
        )

    return resized, True


def prepare_cropped_images():
    """
    为了兼容原来的 UI，函数名仍叫 prepare_cropped_images。
    但这一版不裁剪，只把原图统一成 5472x3648 后送给 PatchCore。

    SAVE_CROPPED_IMAGES = True:
        保存 resize 后的完整原图到 CROPPED_OUTPUT_DIR

    SAVE_CROPPED_IMAGES = False:
        保存到 TEMP_CROPPED_OUTPUT_DIR，预测完成后自动删除
    """
    if SAVE_CROPPED_IMAGES:
        output_dir = CROPPED_OUTPUT_DIR
    else:
        output_dir = TEMP_CROPPED_OUTPUT_DIR

    if CLEAR_CROPPED_DIR and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(RAW_INPUT_PATH)

    if not images:
        print(f"[ERROR] 没有找到原始图片: {RAW_INPUT_PATH}")
        return [], output_dir

    print(f"[INFO] 找到原始图片 {len(images)} 张")
    print(f"[INFO] 预处理输出目录: {output_dir}")
    print(f"[INFO] 是否保留预处理图: {SAVE_CROPPED_IMAGES}")
    print(f"[INFO] 目标尺寸: width={TARGET_WIDTH}, height={TARGET_HEIGHT}")

    processed_paths = []

    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            print(f"[ERROR] 读取失败: {img_path}")
            continue

        processed, resized_flag = resize_full_image_if_needed(img, img_path)

        out_path = output_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), processed)

        processed_paths.append(out_path)

        action = "RESIZE" if resized_flag else "KEEP"
        print(f"[{action} {idx + 1}/{len(images)}] {img_path.name} -> {out_path}")

    return processed_paths, output_dir


# ============================================================
# 9. Anomalib 数据兼容函数
# ============================================================
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


def path_to_list(x):
    if isinstance(x, (list, tuple)):
        return [str(p) for p in x]

    return [str(x)]


def get_batch_attr(batch, *names):
    for name in names:
        if hasattr(batch, name):
            return getattr(batch, name)

        if isinstance(batch, dict) and name in batch:
            return batch[name]

    return None


def anomaly_map_to_list(anomaly_map, batch_count):
    if anomaly_map is None:
        return [None] * batch_count

    if hasattr(anomaly_map, "detach"):
        anomaly_map = anomaly_map.detach().cpu().numpy()

    arr = np.asarray(anomaly_map)

    if arr.ndim == 4:
        arr = arr[:, 0, :, :]

    if arr.ndim == 3:
        if arr.shape[0] == batch_count:
            return [arr[i] for i in range(batch_count)]
        return [arr[0]] * batch_count

    if arr.ndim == 2:
        return [arr] * batch_count

    return [None] * batch_count


def read_gray_image(image_path, target_size=TARGET_SIZE_WH):
    """
    兼容 UI 旧逻辑。
    这版二级规则关闭，一般不会用到。
    target_size 使用 OpenCV 格式：(width, height)。
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img is None:
        return None, None

    if (img.shape[1], img.shape[0]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


# ============================================================
# 10. 构建 PatchCore 模型
# ============================================================
def build_model(pre_trained=False):
    """
    预测端建议 pre_trained=False，防止客户电脑联网下载 timm 预训练权重。
    训练时使用的权重会从 model.ckpt 中加载。
    """
    post_processor = PostProcessor(
        image_sensitivity=IMAGE_SENSITIVITY,
        pixel_sensitivity=PIXEL_SENSITIVITY,
    )

    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pre_trained=pre_trained,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        num_neighbors=NUM_NEIGHBORS,
        post_processor=post_processor,
    )

    return model


# ============================================================
# 11. 主流程：预处理完整原图 + 预测 + 保存 CSV
# ============================================================
def predict_ok_ng(predict_image_dir):
    predict_image_dir = Path(predict_image_dir)

    if not predict_image_dir.exists():
        print(f"[ERROR] 预处理图片目录不存在: {predict_image_dir}")
        return

    dataset = PredictDataset(
        path=predict_image_dir,
        image_size=IMAGE_SIZE,
    )

    model = build_model(pre_trained=False)

    # 清空 PatchCore 热力图 / 结果图目录
    if CLEAR_PATCHCORE_RESULT_DIR and PATCHCORE_RESULT_ROOT.exists():
        shutil.rmtree(PATCHCORE_RESULT_ROOT)

    PATCHCORE_RUN_ROOT.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        default_root_dir=str(PATCHCORE_RUN_ROOT),
        accelerator="gpu",
        devices=1,
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=CKPT_PATH,
        return_predictions=True,
    )

    if predictions is None:
        print("[ERROR] 没有拿到预测结果")
        return

    rows = []

    print("\n========== 最终预测结果 ==========")

    for batch in predictions:
        image_paths = path_to_list(get_batch_attr(batch, "image_path"))
        pred_labels = value_to_list(get_batch_attr(batch, "pred_label"))
        pred_scores = value_to_list(get_batch_attr(batch, "pred_score"))

        for idx, img_path in enumerate(image_paths):
            label_int = int(pred_labels[idx])
            score_float = float(pred_scores[idx])

            if CUSTOM_SCORE_THRESHOLD is None:
                final_result = "NG" if label_int == 1 else "OK"
            else:
                final_result = "NG" if score_float >= CUSTOM_SCORE_THRESHOLD else "OK"

            filename = Path(img_path).name

            print(
                f"{filename:40s}  "
                f"final={final_result:2s}  "
                f"score={score_float:.6f}  "
                f"label={label_int}"
            )

            rows.append({
                "filename": filename,
                "image_path": img_path,
                "final_result": final_result,
                "pred_label": label_int,
                "pred_score": score_float,
            })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "image_path",
                "final_result",
                "pred_label",
                "pred_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n[DONE] 完整原图预处理 + 预测完成")
    print(f"[CSV] {OUT_CSV}")


# ============================================================
# 12. UI 调用入口
# ============================================================
def run_predict_from_ui(
    raw_input_path,
    ckpt_path,
    output_dir,
    save_cropped_images=False,
    clear_cropped_dir=True,
    clear_patchcore_result_dir=True,
    use_stamped_patchcore_dir=True,
    custom_score_threshold=None,
):
    """
    给 PySide6 UI 调用的入口函数。

    raw_input_path:
        单张原图路径或文件夹路径

    ckpt_path:
        PatchCore 的 model.ckpt 路径

    output_dir:
        输出目录，里面会生成 csv、预处理图、热力图目录

    save_cropped_images:
        为了兼容旧 UI，变量名保留。
        True 表示保留 resize 后的完整原图。
        False 表示预测后删除临时预处理图。

    custom_score_threshold:
        None 表示使用 Anomalib 默认 pred_label
        数字表示使用 pred_score 自定义阈值判断 OK/NG
    """

    global RAW_INPUT_PATH
    global CKPT_PATH

    global SAVE_CROPPED_IMAGES
    global CROPPED_OUTPUT_DIR
    global TEMP_CROPPED_OUTPUT_DIR
    global CLEAR_CROPPED_DIR

    global OUT_CSV_BASE
    global OUT_CSV

    global PATCHCORE_RESULT_ROOT
    global CLEAR_PATCHCORE_RESULT_DIR
    global USE_STAMPED_PATCHCORE_DIR
    global PATCHCORE_RUN_ROOT

    global CUSTOM_SCORE_THRESHOLD
    global STAMP

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    RAW_INPUT_PATH = Path(raw_input_path)
    CKPT_PATH = str(ckpt_path)

    SAVE_CROPPED_IMAGES = bool(save_cropped_images)
    CLEAR_CROPPED_DIR = bool(clear_cropped_dir)

    CROPPED_OUTPUT_DIR = output_dir / "processed_full_5472x3648"
    TEMP_CROPPED_OUTPUT_DIR = output_dir / "_temp_full_predict"

    OUT_CSV_BASE = output_dir / "predict.csv"
    OUT_CSV = OUT_CSV_BASE.with_name(
        f"{OUT_CSV_BASE.stem}_{STAMP}{OUT_CSV_BASE.suffix}"
    )

    PATCHCORE_RESULT_ROOT = output_dir / "patchcore_vis"
    CLEAR_PATCHCORE_RESULT_DIR = bool(clear_patchcore_result_dir)
    USE_STAMPED_PATCHCORE_DIR = bool(use_stamped_patchcore_dir)

    if USE_STAMPED_PATCHCORE_DIR:
        PATCHCORE_RUN_ROOT = PATCHCORE_RESULT_ROOT / f"run_{STAMP}"
    else:
        PATCHCORE_RUN_ROOT = PATCHCORE_RESULT_ROOT

    CUSTOM_SCORE_THRESHOLD = custom_score_threshold

    print("========== Step 1: 原图尺寸检查 / resize ==========")
    processed, predict_image_dir = prepare_cropped_images()

    if not processed:
        raise RuntimeError("没有可预测的图片，请检查输入路径。")

    print("\n========== Step 2: PatchCore OK/NG 预测 ==========")
    predict_ok_ng(predict_image_dir)

    if not SAVE_CROPPED_IMAGES:
        try:
            if predict_image_dir.exists():
                shutil.rmtree(predict_image_dir)
                print(f"[INFO] 已删除临时预处理目录: {predict_image_dir}")
        except Exception as e:
            print(f"[WARN] 删除临时预处理目录失败: {predict_image_dir}, err={e}")

    if not OUT_CSV.exists():
        raise RuntimeError(f"预测结束但没有找到 CSV 文件: {OUT_CSV}")

    return str(OUT_CSV)


# ============================================================
# 13. 命令行入口
# ============================================================
def main():
    csv_path = run_predict_from_ui(
        raw_input_path=RAW_INPUT_PATH,
        ckpt_path=CKPT_PATH,
        output_dir=BASE_DIR / "output",
        save_cropped_images=SAVE_CROPPED_IMAGES,
        clear_cropped_dir=CLEAR_CROPPED_DIR,
        clear_patchcore_result_dir=CLEAR_PATCHCORE_RESULT_DIR,
        use_stamped_patchcore_dir=USE_STAMPED_PATCHCORE_DIR,
        custom_score_threshold=CUSTOM_SCORE_THRESHOLD,
    )

    print(f"[DONE] CSV: {csv_path}")


if __name__ == "__main__":
    main()
