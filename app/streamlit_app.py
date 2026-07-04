from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import (  # noqa: E402
    CLASS_COLORS, CLASS_TITLES, analyze_image, build_model, make_confidence_map,
    make_overlay, make_pdf_report, metrics_frame, tiles_geojson,
)
from src.segmentation import infer as seg  # noqa: E402

st.set_page_config(page_title="Ore Vision · Анализ шлифов", page_icon="◉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'Manrope',sans-serif}
.stApp {background:radial-gradient(circle at 85% 0%,#17332b 0,#0a1514 28%,#07100f 70%);color:#eef7f2}
[data-testid="stSidebar"] {background:#0c1917;border-right:1px solid #203b35}
/* High-contrast typography on the dark canvas. */
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stWidgetLabel"] p {color:#c8d9d3}
.stApp h1,.stApp h2,.stApp h3,.stApp h4 {color:#f4fbf7}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {color:#d3e2dd !important}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {color:#f5fbf8 !important}
/* Streamlit renders secondary/download buttons on a light surface. */
.stApp button[kind="secondary"],
.stApp [data-testid="stBaseButton-secondary"],
.stApp [data-testid="stDownloadButton"] button {background:#f1f7f4;border-color:#bfd2ca;color:#10231e !important}
.stApp button[kind="secondary"] p,
.stApp [data-testid="stBaseButton-secondary"] p,
.stApp [data-testid="stDownloadButton"] button p {color:#10231e !important;font-weight:700}
.stApp button[kind="secondary"]:hover,
.stApp [data-testid="stBaseButton-secondary"]:hover,
.stApp [data-testid="stDownloadButton"] button:hover {background:#dff2e9;border-color:#64d39a;color:#091713 !important}
/* Controls keep readable values and hints. */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {background:#132622;border-color:#345047;color:#eef7f2}
[data-baseweb="select"] span,
[data-baseweb="input"] input {color:#eef7f2 !important}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {color:#bfd2ca !important}
.hero {padding:1.2rem 0 1.8rem}.eyebrow {color:#64d39a;font-weight:800;letter-spacing:.16em;font-size:.72rem}
.hero h1 {font-size:clamp(2.3rem,5vw,4.6rem);line-height:.96;margin:.55rem 0;color:#f4fbf7;letter-spacing:-.055em}
.hero p {max-width:760px;color:#9eb5ad;font-size:1.05rem}
.pill {display:inline-block;padding:.32rem .68rem;border:1px solid #2d5549;border-radius:999px;color:#a9c5bc;margin-right:.35rem;font-size:.78rem}
.result-card {padding:1.25rem 1.4rem;border-radius:18px;background:linear-gradient(135deg,#173b30,#10241f);border:1px solid #34705c;margin:.7rem 0 1.2rem}
.result-card small {color:#80ab9c}.result-card h2 {margin:.25rem 0;color:#f5fff9}.result-card p {color:#b3c9c1;margin:0}
[data-testid="stMetric"] {background:#10201d;border:1px solid #244039;padding:1rem;border-radius:14px}
[data-testid="stFileUploaderDropzone"] {background:#10201d;border:1px dashed #43846d}
.legend {display:flex;gap:1rem;flex-wrap:wrap;color:#afc6be;font-size:.8rem;margin:.2rem 0 1rem}
.dot {width:.65rem;height:.65rem;display:inline-block;border-radius:50%;margin-right:.3rem}
.note {padding:.8rem 1rem;border-left:3px solid #eab308;background:#241f0d;color:#e9dca0;border-radius:0 10px 10px 0;font-size:.83rem}
</style>
""", unsafe_allow_html=True)


MODEL_LABELS = {"ResNet18 · лучший F1": "resnet18", "MobileNetV3 · быстрый": "mobilenet_v3_small", "TinyCNN · лёгкий": "tinycnn"}


@st.cache_data(show_spinner=False)
def load_ml_summary() -> dict:
    """Read presentation metrics from training artifacts instead of hardcoding them."""
    summary = {
        "classifier_count": len(MODEL_LABELS),
        "total_models": len(MODEL_LABELS),
        "best_val_f1": None,
        "best_classifier": None,
        "test_f1": None,
        "test_size": None,
        "unet_dice": None,
        "unet_iou": None,
    }
    display_names = {
        "resnet18": "ResNet18",
        "mobilenet_v3_small": "MobileNetV3",
        "tinycnn": "TinyCNN",
    }
    candidates = []
    for model_name in display_names:
        history_path = ROOT / "reports" / f"{model_name}_history.csv"
        if not history_path.exists():
            continue
        history = pd.read_csv(history_path)
        if "val_macro_f1" in history and history["val_macro_f1"].notna().any():
            candidates.append((float(history["val_macro_f1"].max()), display_names[model_name]))
    if candidates:
        summary["best_val_f1"], summary["best_classifier"] = max(candidates)

    predictions_path = ROOT / "reports" / "resnet18_test_predictions.csv"
    if predictions_path.exists():
        predictions = pd.read_csv(predictions_path, usecols=["label", "pred_label"])
        class_f1 = []
        for label in sorted(predictions["label"].dropna().unique()):
            actual = predictions["label"] == label
            predicted = predictions["pred_label"] == label
            tp = int((actual & predicted).sum())
            fp = int((~actual & predicted).sum())
            fn = int((actual & ~predicted).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            class_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        if class_f1:
            summary["test_f1"] = sum(class_f1) / len(class_f1)
            summary["test_size"] = len(predictions)

    unet_history_path = ROOT / "reports" / "unet_talc_history.csv"
    if unet_history_path.exists():
        history = pd.read_csv(unet_history_path)
        if "val_dice" in history and history["val_dice"].notna().any():
            summary["unet_dice"] = float(history["val_dice"].max())
            summary["total_models"] += 1
        if "val_iou" in history and history["val_iou"].notna().any():
            summary["unet_iou"] = float(history["val_iou"].max())
    return summary


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    path = ROOT / "models" / f"{model_name}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Не найдены веса модели: {path}")
    return build_model(model_name, path)


@st.cache_resource(show_spinner=False)
def load_talc_segmenter():
    """Load the standalone talc U-Net (separate model from the classifier)."""
    path = ROOT / "models" / "unet_talc.pth"
    if not path.exists():
        return None
    return seg.load_talc_model(path)


def read_image(uploaded) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    source = Image.open(BytesIO(uploaded.getvalue()))
    original_size = source.size
    max_pixels = 30_000_000
    if source.width * source.height > max_pixels:
        scale = (max_pixels / (source.width * source.height)) ** 0.5
        target = (max(1, int(source.width * scale)), max(1, int(source.height * scale)))
        # JPEG can decode directly near the target resolution and avoid a huge
        # temporary RGB allocation. Other formats fall back to thumbnail below.
        source.draft("RGB", target)
    image = source.convert("RGB")
    if image.width * image.height > max_pixels:
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        image.thumbnail((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    image.info["ore_original_size"] = original_size
    return image


def result_text(result: dict) -> str:
    label = CLASS_TITLES[result["final_label"]]
    return (f"Руда классифицирована как {label.lower()}: доля тайлов с признаками талька — "
            f"{result['shares']['talc']:.1%}, тонких срастаний — {result['fine_share']:.1%}, "
            f"средняя уверенность — {result['mean_confidence']:.1%}.")


with st.sidebar:
    st.markdown("### Параметры анализа")
    model_label = st.selectbox("Модель", list(MODEL_LABELS))
    tile_size = st.select_slider("Размер тайла", options=[256, 384, 512, 768, 1024], value=512)
    overlap = st.slider("Перекрытие тайлов", 0, min(192, tile_size // 2), min(64, tile_size // 4), 16)
    talc_threshold = st.slider("Порог оталькованной руды", 1, 40, 10, 1) / 100
    max_tiles = st.slider("Максимум тайлов", 25, 500, 250, 25)
    st.divider()
    st.caption("Локальная обработка · данные не покидают рабочую станцию")
    st.caption("Цвета: зелёный — обычные, красный — тонкие, синий — тальк")

st.markdown("""
<div class="hero">
  <div class="eyebrow">NORNICKEL HACKATHON · GEOLOGY AI</div>
  <h1>Скажи мне,<br>кто твой шлиф</h1>
  <p>Интерпретируемый анализ панорамных OM-изображений: классификация руды, карта тайлов, оценка фаз и готовый лабораторный отчёт.</p>
  <span class="pill">TIFF · PNG · JPEG</span><span class="pill">локальный inference</span><span class="pill">экспорт CSV / PDF</span>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Перетащите один или несколько снимков шлифа",
    type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True,
)

if not uploaded_files:
    ml = load_ml_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ML-модели", str(ml["total_models"]), "3 классификатора + U-Net")
    col2.metric(
        "Лучшая val macro-F1",
        f'{ml["best_val_f1"]:.1%}' if ml["best_val_f1"] is not None else "—",
        ml["best_classifier"] or "нет отчёта",
    )
    col3.metric(
        "Test macro-F1",
        f'{ml["test_f1"]:.1%}' if ml["test_f1"] is not None else "—",
        f'ResNet18 · {ml["test_size"]} снимков' if ml["test_size"] else "нет отчёта",
    )
    col4.metric(
        "Лучший val Dice",
        f'{ml["unet_dice"]:.1%}' if ml["unet_dice"] is not None else "—",
        f'U-Net тальк · IoU {ml["unet_iou"]:.1%}' if ml["unet_iou"] is not None else "сегментатор",
    )
    st.caption(
        "Классификация: ResNet18, MobileNetV3 Small и TinyCNN. "
        "Пиксельная сегментация талька: ResNet18 U-Net; сульфидные фазы: классический CV."
    )
    st.info("Загрузите снимок — параметры можно настроить слева. Для демо достаточно обычного JPG.")
    st.stop()

try:
    model_name = MODEL_LABELS[model_label]
    model = load_model(model_name)
except Exception as exc:
    st.error(f"Модель не загрузилась: {exc}")
    st.stop()

batch_rows = []
for file_index, uploaded in enumerate(uploaded_files):
    with st.container(border=True):
        st.markdown(f"### {uploaded.name}")
        try:
            image = read_image(uploaded)
        except Exception as exc:
            st.error(f"Не удалось прочитать изображение: {exc}")
            continue

        original_size = image.info.get("ore_original_size", image.size)
        if original_size != image.size:
            st.warning(
                f"Большая панорама безопасно уменьшена с {original_size[0]}×{original_size[1]} "
                f"до {image.width}×{image.height} для облачного анализа."
            )

        progress = st.progress(0, text="Анализируем тайлы…")
        try:
            result = analyze_image(
                model, image, tile_size=tile_size, overlap=overlap, max_tiles=max_tiles,
                talc_threshold=talc_threshold, progress=lambda value: progress.progress(value, text="Анализируем тайлы…"),
            )
        except Exception as exc:
            progress.empty()
            st.error(f"Ошибка inference: {exc}")
            continue
        progress.empty()

        overlay = make_overlay(image, result["tiles"])
        confidence_map = make_confidence_map(image, result["tiles"])
        conclusion = result_text(result)
        st.markdown(
            f'<div class="result-card"><small>ИТОГОВАЯ КЛАССИФИКАЦИЯ</small><h2>{CLASS_TITLES[result["final_label"]]}</h2><p>{conclusion}</p></div>',
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Тальк", f'{result["shares"]["talc"]:.1%}', f'порог {talc_threshold:.0%}')
        metric_cols[1].metric("Тонкие срастания", f'{result["fine_share"]:.1%}')
        metric_cols[2].metric("Сульфидные области", f'{result["sulfide_share"]:.1%}')
        metric_cols[3].metric("Уверенность", f'{result["mean_confidence"]:.1%}', f'{result["tile_count"]} тайлов')

        tab_seg, tab_map, tab_conf, tab_metrics, tab_tiles = st.tabs(
            ["Пиксельная сегментация", "Карта классов (тайлы)", "Карта уверенности", "Метрики", "Тайлы"]
        )
        with tab_seg:
            segmenter = load_talc_segmenter()
            if segmenter is None:
                st.warning(
                    "Веса сегментатора не найдены (models/unet_talc.pth). "
                    "Обучите модель: `python -m src.segmentation.train_talc`."
                )
            else:
                with st.spinner("Сегментация фаз (U-Net тальк + CV сульфиды)…"):
                    seg_result = seg.segment(
                        np.asarray(image), talc_model=segmenter, ore_talc_threshold=talc_threshold,
                    )
                seg_overlay = seg.overlay(seg_result["work_image"], seg_result["labels"])
                seg_flat = Image.fromarray(seg.colored_mask(seg_result["labels"]))
                st.markdown(
                    '<div class="legend">'
                    '<span><i class="dot" style="background:#22c55e"></i>обычные срастания</span>'
                    '<span><i class="dot" style="background:#ef4444"></i>тонкие срастания</span>'
                    '<span><i class="dot" style="background:#3b82f6"></i>тальк</span></div>',
                    unsafe_allow_html=True,
                )
                seg_left, seg_right = st.columns(2)
                seg_left.image(seg_overlay, caption="Сегментация поверх снимка", use_container_width=True)
                seg_right.image(seg_flat, caption="Маска фаз (пиксели)", use_container_width=True)
                seg_metric_cols = st.columns(4)
                seg_metric_cols[0].metric("Тальк (U-Net)", f'{seg_result["talc_share"]:.1%}')
                seg_metric_cols[1].metric("Сульфиды", f'{seg_result["sulfide_share"]:.1%}')
                seg_metric_cols[2].metric("Тонкие срастания", f'{seg_result["fine_share"]:.1%}')
                seg_metric_cols[3].metric("Обычные срастания", f'{seg_result["ordinary_share"]:.1%}')
                st.caption(seg.result_text(seg_result))
                seg_frame = pd.DataFrame(seg.metrics_rows(seg_result), columns=["Метрика", "Значение", "Метод"])
                st.dataframe(seg_frame, hide_index=True, use_container_width=True)
                seg_buffer = BytesIO(); seg_overlay.save(seg_buffer, format="PNG")
                st.download_button(
                    "Скачать сегментацию (PNG)", seg_buffer.getvalue(),
                    f"{Path(uploaded.name).stem}_segmentation.png", "image/png",
                    key=f"seg-{file_index}", use_container_width=True,
                )
        with tab_map:
            st.markdown('<div class="legend"><span><i class="dot" style="background:#22c55e"></i>обычные</span><span><i class="dot" style="background:#ef4444"></i>тонкие</span><span><i class="dot" style="background:#3b82f6"></i>тальк</span></div>', unsafe_allow_html=True)
            left, right = st.columns(2)
            left.image(image, caption="Исходное изображение", width="stretch")
            right.image(overlay, caption="Интерпретируемая tile-карта", width="stretch")
        with tab_conf:
            st.image(confidence_map, caption="Ярче — ниже уверенность модели", width="stretch")
        with tab_metrics:
            st.dataframe(metrics_frame(result), hide_index=True, width="stretch")
            st.markdown('<div class="note">Доли талька и типов срастаний — оценки по классифицированным тайлам. Для точных пиксельных процентов потребуется отдельная сегментационная модель.</div>', unsafe_allow_html=True)
        with tab_tiles:
            tile_view = result["tiles"].copy()
            tile_view.insert(6, "corrected_label", tile_view["pred_label"])
            editable_view = tile_view.drop(columns=["area", "sulfide_proxy"])
            edited = st.data_editor(
                editable_view, hide_index=True,
                disabled=[column for column in editable_view.columns if column != "corrected_label"],
                column_config={
                    "corrected_label": st.column_config.SelectboxColumn(
                        "Экспертная метка", options=["ordinary", "difficult", "talc"], required=True,
                    ),
                    "confidence": st.column_config.ProgressColumn("Уверенность", min_value=0.0, max_value=1.0, format="%.2f"),
                },
                width="stretch", key=f"editor-{file_index}",
            )
            corrections = edited[edited.corrected_label != edited.pred_label].copy()
            if len(corrections):
                corrections.insert(0, "source_file", uploaded.name)
                st.success(f"Отмечено исправлений: {len(corrections)}. Их можно добавить в active-learning набор.")
                st.download_button(
                    "Скачать экспертные исправления", corrections.to_csv(index=False).encode("utf-8-sig"),
                    f"{Path(uploaded.name).stem}_corrections.csv", "text/csv", key=f"corrections-{file_index}",
                )

        export_cols = st.columns(4)
        metrics_csv = metrics_frame(result).to_csv(index=False).encode("utf-8-sig")
        export_cols[0].download_button("Скачать CSV", metrics_csv, f"{Path(uploaded.name).stem}_metrics.csv", "text/csv", key=f"csv-{file_index}", width="stretch")
        pdf = make_pdf_report(uploaded.name, result, overlay)
        export_cols[1].download_button("Скачать PDF", pdf, f"{Path(uploaded.name).stem}_report.pdf", "application/pdf", key=f"pdf-{file_index}", width="stretch")
        overlay_buffer = BytesIO(); overlay.save(overlay_buffer, format="PNG")
        export_cols[2].download_button("Скачать карту", overlay_buffer.getvalue(), f"{Path(uploaded.name).stem}_overlay.png", "image/png", key=f"map-{file_index}", width="stretch")
        geojson = tiles_geojson(result["tiles"], image.size)
        export_cols[3].download_button("Скачать GeoJSON", geojson, f"{Path(uploaded.name).stem}_tiles.geojson", "application/geo+json", key=f"geo-{file_index}", width="stretch")
        st.caption("GeoJSON использует координаты пикселей изображения; CRS и физический масштаб добавляются только при наличии метаданных съёмки.")

        batch_rows.append({
            "file": uploaded.name, "result": CLASS_TITLES[result["final_label"]],
            "talc_percent": round(result["shares"]["talc"] * 100, 2),
            "ordinary_percent": round(result["common_share"] * 100, 2),
            "difficult_percent": round(result["fine_share"] * 100, 2),
            "sulfide_percent": round(result["sulfide_share"] * 100, 2),
            "confidence_percent": round(result["mean_confidence"] * 100, 2),
            "tiles": result["tile_count"], "conclusion": conclusion,
        })

if len(batch_rows) > 1:
    st.markdown("## Сводка партии")
    batch = pd.DataFrame(batch_rows)
    st.dataframe(batch, hide_index=True, width="stretch")
    st.download_button("Скачать сводный CSV", batch.to_csv(index=False).encode("utf-8-sig"), "ore_batch_results.csv", "text/csv")
