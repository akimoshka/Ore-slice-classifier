from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

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


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    path = ROOT / "models" / f"{model_name}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Не найдены веса модели: {path}")
    return build_model(model_name, path)


def read_image(uploaded) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(BytesIO(uploaded.getvalue())).convert("RGB")


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
    col1, col2, col3 = st.columns(3)
    col1.metric("Классы", "3", "рядовая · трудная · тальк")
    col2.metric("Лучшая val F1", "87.2%", "ResNet18")
    col3.metric("Режим", "Tile-based", "для панорам")
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

        tab_map, tab_conf, tab_metrics, tab_tiles = st.tabs(["Карта классов", "Карта уверенности", "Метрики", "Тайлы"])
        with tab_map:
            st.markdown('<div class="legend"><span><i class="dot" style="background:#22c55e"></i>обычные</span><span><i class="dot" style="background:#ef4444"></i>тонкие</span><span><i class="dot" style="background:#3b82f6"></i>тальк</span></div>', unsafe_allow_html=True)
            left, right = st.columns(2)
            left.image(image, caption="Исходное изображение", use_container_width=True)
            right.image(overlay, caption="Интерпретируемая tile-карта", use_container_width=True)
        with tab_conf:
            st.image(confidence_map, caption="Ярче — ниже уверенность модели", use_container_width=True)
        with tab_metrics:
            st.dataframe(metrics_frame(result), hide_index=True, use_container_width=True)
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
                use_container_width=True, key=f"editor-{file_index}",
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
        export_cols[0].download_button("Скачать CSV", metrics_csv, f"{Path(uploaded.name).stem}_metrics.csv", "text/csv", key=f"csv-{file_index}", use_container_width=True)
        pdf = make_pdf_report(uploaded.name, result, overlay)
        export_cols[1].download_button("Скачать PDF", pdf, f"{Path(uploaded.name).stem}_report.pdf", "application/pdf", key=f"pdf-{file_index}", use_container_width=True)
        overlay_buffer = BytesIO(); overlay.save(overlay_buffer, format="PNG")
        export_cols[2].download_button("Скачать карту", overlay_buffer.getvalue(), f"{Path(uploaded.name).stem}_overlay.png", "image/png", key=f"map-{file_index}", use_container_width=True)
        geojson = tiles_geojson(result["tiles"], image.size)
        export_cols[3].download_button("Скачать GeoJSON", geojson, f"{Path(uploaded.name).stem}_tiles.geojson", "application/geo+json", key=f"geo-{file_index}", use_container_width=True)
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
    st.dataframe(batch, hide_index=True, use_container_width=True)
    st.download_button("Скачать сводный CSV", batch.to_csv(index=False).encode("utf-8-sig"), "ore_batch_results.csv", "text/csv")
