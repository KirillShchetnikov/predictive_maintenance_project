import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Предобработка данных
        # Удаляем ненужные столбцы
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        # Преобразуем категориальные переменные 'Type' и 'Product ID' в числовую
        if 'Type' in data.columns:
            data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Разделение данных на признаки (X) и целевую переменную (y)
        # В задании целевая переменная называется "Target"
        if 'Machine failure' in data.columns:
            X = data.drop(columns=['Machine failure'])
            y = data['Machine failure']
        else:
            st.error("В загруженном датасете не найден заданный столбец")
            return

        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Обучение модели
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Оценка модели
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        # Визуализация результатов
        st.header("Результаты обучения модели")
        st.write(f"Accuracy: {accuracy:.2f}")

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_rep)

        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            # Здесь предполагается, что модель обучалась на числовых данных,
            # поэтому вводимые пользователем значения должны соответствовать обучающему набору.
            # Если среди признаков есть категориальные, их тоже нужно преобразовывать.
            # Пример для числовых признаков:
            air_temp = st.number_input("Air temperature [K]", value=300.0)
            process_temp = st.number_input("Process temperature [K]", value=310.0)
            rotational_speed = st.number_input("Rotational speed [rpm]", value=1500)
            torque = st.number_input("Torque [Nm]", value=40.0)
            tool_wear = st.number_input("Tool wear [min]", value=10)
            type = st.number_input("Type", value=1)

            submit_button = st.form_submit_button("Предсказать")
            if submit_button:
                # Преобразование введенных данных в DataFrame
                input_data = pd.DataFrame({
                    'Type': [type],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotational_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear]
                    # Если в обучении использовались и другие признаки (например, 'Type'),
                    # необходимо также добавить их, возможно, задать значение по умолчанию или выбрать из интерфейса.
                })
                # Важно: если в обучающих данных был проведён масштабирование или другая трансформация,
                # то её нужно применить и к input_data.

                # Предсказание
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[:, 1]
                st.write(f"Предсказание: {prediction[0]}")
                st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")


analysis_and_model_page()