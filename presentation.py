import streamlit as st
import reveal_slides as rs
import base64
import os

def presentation_page():
    st.title("Презентация проекта")

    # Список изображений для каждого слайда
    image_names = [
        "step1.jpg",  # Для "Загрузка данных"
        "step5.jpg"   # Для "Визуализация результатов"
    ]

    # Загрузка изображений и преобразование в base64
    images_base64 = {}
    for img_name in image_names:
        image_path = os.path.join("data", img_name)
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                images_base64[img_name] = base64.b64encode(image_data).decode("utf-8")
        except FileNotFoundError:
            st.warning(f"Изображение {image_path} не найдено, слайд будет без фото.")
            images_base64[img_name] = None

    # Теперь формируем содержимое презентации
    presentation_markdown = "# Прогнозирование отказов оборудования\n"

    # Введение
    presentation_markdown += """
---
## Введение
- Задача проекта: разработать модель машинного обучения для бинарной классификации.
- Используется синтетический набор "AI4I 2020 Predictive Maintenance Dataset" (10 000 записей, 14 признаков).
- Цель: предсказать отказ оборудования (1) или его отсутствие (0).
"""

    # Этапы работы
    steps = [
        "1. Загрузка данных\n",
        "2. Предобработка данных",
        "3. Обучение модели",
        "4. Оценка модели",
        "5. Визуализация результатов\n"
    ]

    descriptions = [
        "Пользователь выбирает файл датасета с компьютера.\n",
        "Удаляем ненужные столбцы (UDI, Product ID, TWF, HDF, PWF, OSF, RNF), преобразуем Type в числовую, проверяем пропуски.",
        "Выбираем 4 модели и определяем лучшую в задаче бинарной классификации.",
        "Используем Accuracy, Confusion Matrix, Classification Report и ROC-AUC для оценки.",
        "Результаты для каждой модели представлены как на фото.\n"
    ]

    for i, (step, desc) in enumerate(zip(steps, descriptions)):
        image_html = ""
        # Для слайда 1
        if i == 0:
            if images_base64["step1.jpg"]:
                image_html = f'<img src="data:image/jpeg;base64,{images_base64["step1.jpg"]}" width="400">'
            else:
                image_html = "Изображение отсутствует"
        # Для слайда 5
        elif i == 4:
            if images_base64["step5.jpg"]:
                image_html = f'<img src="data:image/jpeg;base64,{images_base64["step5.jpg"]}" width="400">'
            else:
                image_html = "Изображение отсутствует"

        presentation_markdown += f"""
---
## {step}
{desc}
{image_html}
"""

    # Остальные слайды
    presentation_markdown += """
---
## Streamlit-приложение
Streamlit-приложение состоит из двух страниц
- Основная страница: анализ данных и предсказания.
- Страница с презентацией: описание проекта.

---
## Заключение
-Итоги работы:\n
Создано интерактивное приложение Streamlit, объединяющее загрузку и анализ данных, обучение моделей, визуализацию результатов и предсказания по новым данным.\n
-Возможные улучшения:\n
Применение дополнительных алгоритмов, углублённая предобработка и оптимизация гиперпараметров, а также расширение функционала визуализации.\n
"""

    # Настройки презентации
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige",
                                      "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave",
                                              "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex",
                                            "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    # Отображение презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )

presentation_page()
