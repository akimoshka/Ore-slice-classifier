import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Ore Slice Classifier",
    layout="wide"
)

st.title("Ore Slice Classifier")
st.caption("ML/CV система для анализа полированных шлифов руды")

st.sidebar.header("Настройки анализа")

talс_threshold = st.sidebar.slider(
    "Порог талька для оталькованной руды (%)",
    min_value=0,
    max_value=50,
    value=10
)

uploaded_file = st.file_uploader(
    "Загрузите изображение шлифа",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("Исходное изображение")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original image", use_container_width=True)

    # Заглушка для будущей модели
    h, w, _ = image_np.shape
    mask = np.zeros_like(image_np)

    # temporary fake mask for demo
    mask[:, :, 1] = 80  # green channel

    overlay = (image_np * 0.7 + mask * 0.3).astype(np.uint8)

    with col2:
        st.image(overlay, caption="Predicted mask overlay", use_container_width=True)

    st.subheader("Метрики анализа")

    # Пока тестовые значения, потом заменим на output модели
    talc_percent = 14.0
    common_sulfides = 38.0
    fine_sulfides = 62.0
    total_sulfides = 21.5

    metrics = pd.DataFrame({
        "Метрика": [
            "Общая доля сульфидов",
            "Обычные срастания",
            "Тонкие срастания",
            "Доля талька"
        ],
        "Значение": [
            f"{total_sulfides}%",
            f"{common_sulfides}%",
            f"{fine_sulfides}%",
            f"{talc_percent}%"
        ]
    })

    st.dataframe(metrics, use_container_width=True)

    st.subheader("Итоговая классификация")

    if talc_percent > talс_threshold:
        result = "Оталькованная руда"
    elif common_sulfides >= fine_sulfides:
        result = "Рядовая руда"
    else:
        result = "Труднообогатимая руда"

    st.success(
        f"Руда классифицирована как: **{result}**. "
        f"Содержание талька — {talc_percent}%, "
        f"преобладание тонких срастаний — {fine_sulfides}%."
    )

    csv = metrics.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Скачать метрики CSV",
        data=csv,
        file_name="ore_analysis_metrics.csv",
        mime="text/csv"
    )

else:
    st.info("Загрузите изображение шлифа, чтобы начать анализ.")