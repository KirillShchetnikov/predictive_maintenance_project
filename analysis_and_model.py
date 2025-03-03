import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Первые строки загруженных данных:")
        st.dataframe(data.head())

        # Предобработка данных
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
        if 'Type' in data.columns:
            data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Переименование столбцов для удаления квадратных скобок
        data.columns = [col.replace('[', '').replace(']', '').replace(' ', '_') for col in data.columns]

        if 'Machine_failure' in data.columns:
            X = data.drop(columns=['Machine_failure'])
            y = data['Machine_failure']
        else:
            st.error("В загруженном датасете не найден столбец 'Machine failure'")
            return

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Инициализация моделей
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False,
                                     eval_metric='logloss'),
            "SVM": SVC(kernel='linear', probability=True, random_state=42)
        }

        # Обучение и оценка моделей
        results = {}
        roc_data = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Вероятности для ROC-AUC

            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

            results[name] = {
                "Accuracy": accuracy,
                "Confusion Matrix": conf_matrix,
                "Classification Report": class_report,
                "ROC-AUC": roc_auc
            }
            roc_data[name] = (fpr, tpr, roc_auc)

        # Визуализация результатов
        st.header("Результаты обучения моделей")

        # Таблица с метриками
        st.subheader("Сравнение метрик")
        metrics_df = pd.DataFrame({
            "Model": results.keys(),
            "Accuracy": [results[name]["Accuracy"] for name in results],
            "ROC-AUC": [results[name]["ROC-AUC"] for name in results]
        })
        st.table(metrics_df)

        # Подробные результаты для каждой модели
        for name, result in results.items():
            st.subheader(f"{name}")
            st.write(f"Accuracy: {result['Accuracy']:.2f}")
            st.write(f"ROC-AUC: {result['ROC-AUC']:.2f}")
            st.text(f"Classification Report:\n{result['Classification Report']}")
            fig, ax = plt.subplots()
            sns.heatmap(result["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        # График ROC-кривых
        st.subheader("ROC-кривые")
        fig, ax = plt.subplots()
        for name, (fpr, tpr, roc_auc) in roc_data.items():
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        # ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
        ax.set_xlabel('Доля ложноположительных предсказаний')
        ax.set_ylabel('Доля истинноположительных предсказаний')
        ax.set_title('ROC-кривые для всех моделей')
        ax.legend()
        st.pyplot(fig)

        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков для предсказания:")
            air_temp = st.number_input("Air temperature (K)", value=300.0)  # Убраны скобки
            process_temp = st.number_input("Process temperature (K)", value=310.0)
            rotational_speed = st.number_input("Rotational speed (rpm)", value=1500)
            torque = st.number_input("Torque (Nm)", value=40.0)
            tool_wear = st.number_input("Tool wear (min)", value=10)
            type_input = st.selectbox("Type", ["L", "M", "H"])
            type_encoded = LabelEncoder().fit_transform([type_input])[0]

            submit_button = st.form_submit_button("Предсказать")
            if submit_button:
                input_data = pd.DataFrame({
                    'Type': [type_encoded],
                    'Air_temperature_K': [air_temp],  # Новые имена без скобок
                    'Process_temperature_K': [process_temp],
                    'Rotational_speed_rpm': [rotational_speed],
                    'Torque_Nm': [torque],
                    'Tool_wear_min': [tool_wear]
                })
                input_data = input_data[X.columns]  # Упорядочиваем столбцы

                # Предсказание для каждой модели
                st.subheader("Результаты предсказания")
                for name, model in models.items():
                    prediction = model.predict(input_data)
                    prediction_proba = model.predict_proba(input_data)[:, 1]
                    st.write(f"{name}:")
                    st.write(f"Предсказание: {prediction[0]}")
                    st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")


analysis_and_model_page()