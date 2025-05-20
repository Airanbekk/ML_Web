from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import logging
logging.basicConfig(level=logging.DEBUG)
import joblib
import pandas as pd
import io
import numpy as np # Добавим для работы с примерами данных (если понадобится, но в этом коде не используется напрямую)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Загружаем модель и скейлер при старте приложения
try:
    model = joblib.load("logreg_model_v2.pkl")
    scaler = joblib.load("scaler_v2.pkl")
    print("Модель и скейлер успешно загружены.")
except FileNotFoundError:
    raise RuntimeError("Не найдены файлы модели или скейлера. Убедитесь, что 'logreg_model_v2.pkl' и 'scaler_v2.pkl' находятся в той же директории, что и приложение FastAPI.")
except Exception as e:
    raise RuntimeError(f"Ошибка при загрузке модели или скейлера: {e}")

# Определяем правильный порядок и названия колонок
# ЭТОТ СПИСОК ДОЛЖЕН ТОЧНО СООТВЕТСТВОВАТЬ feature_names из sklearn.datasets.load_breast_cancer().
# В НЁМ ДОЛЖНО БЫТЬ 30 ПРИЗНАКОВ И НЕ ДОЛЖНО БЫТЬ 'target'.
correct_columns = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

def process_uploaded_file(file_content: bytes) -> pd.DataFrame:
    """
    Обрабатывает загруженный CSV-файл.
    Явно удаляет колонку 'target', если она присутствует.
    Проверяет и переупорядочивает столбцы, а также проверяет на ошибки данных.
    """
    s = io.StringIO(file_content.decode('utf-8'))
    
    df = None
    # Пытаемся прочитать CSV с разными разделителями
    print("Попытка чтения CSV с разными разделителями...")
    for sep_char in [',', ';', '\t', ' ']: # Попробуем запятую, точку с запятой, табуляцию, пробел
        try:
            # Для пробела используем sep='\s+' чтобы учесть несколько пробелов
            if sep_char == ' ':
                df = pd.read_csv(s, sep='\s+', engine='python') # engine='python' для sep='\s+'
            else:
                df = pd.read_csv(s, sep=sep_char)
            print(f"CSV успешно прочитан с разделителем: '{sep_char}'.")
            break # Если успешно прочитали, выходим из цикла
        except pd.errors.ParserError:
            s.seek(0) # Сброс указателя файла в начало для следующей попытки
            print(f"Не удалось прочитать с разделителем '{sep_char}'. Пробую следующий.")
            continue
    
    if df is None:
        raise ValueError("Не удалось прочитать CSV-файл после попыток с разными разделителями. Проверьте формат файла.")

    print("\n--- Исходный DataFrame после чтения CSV (до обработки): ---")
    print(df.head())
    print(f"Исходное количество колонок: {df.shape[1]}")
    print("Типы данных в исходном DataFrame:")
    print(df.info())

    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: ЯВНО УДАЛЯЕМ КОЛОНКУ 'target', если она есть ---
    # Делаем это ДО проверки correct_columns, чтобы 'df' содержал только признаки.
    # Проверяем на разные варианты написания 'target'
    cols_to_drop = [col for col in df.columns if col.strip().lower() == 'target']
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Колонка(и) '{cols_to_drop}' успешно удалена(ы) из DataFrame.")
    else:
        print("Колонка 'target' не найдена и не удалена.")

    # Проверяем, что все ожидаемые столбцы (из correct_columns) присутствуют в загруженном файле
    # Теперь 'df' должен содержать только признаки.
    missing_columns = [col for col in correct_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют необходимые столбцы в файле: {', '.join(missing_columns)}. Пожалуйста, проверьте формат файла. Ожидаемые: {', '.join(correct_columns)}")

    # Выбираем и переупорядочиваем столбцы в соответствии с correct_columns
    df_ordered = df[correct_columns]

    print("\n--- DataFrame после удаления 'target' и переупорядочивания (должно быть 30 признаков): ---")
    print(df_ordered.head())
    print(f"Количество колонок после обработки: {df_ordered.shape[1]}")

    # Дополнительная проверка на количество признаков
    expected_count = len(correct_columns) # Это всегда 30
    if df_ordered.shape[1] != expected_count:
        raise ValueError(f"Количество столбцов ({df_ordered.shape[1]}) не соответствует ожидаемому ({expected_count}) после обработки. Возможно, проблема с названиями колонок или их количеством.")

    # Проверка на пропущенные значения
    if df_ordered.isnull().sum().sum() > 0:
        raise ValueError("Файл содержит пропущенные значения (NaN). Пожалуйста, уберите их или заполните перед загрузкой.")
    
    # Проверка, что все колонки - числовые
    if not all(pd.api.types.is_numeric_dtype(df_ordered[col]) for col in df_ordered.columns):
        non_numeric_cols = [col for col in df_ordered.columns if not pd.api.types.is_numeric_dtype(df_ordered[col])]
        raise ValueError(f"Найдены нечисловые значения в колонках: {', '.join(non_numeric_cols)}. Убедитесь, что все данные числовые.")

    return df_ordered

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    """
    Отображает форму для загрузки CSV-файла.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Принимает загруженный CSV-файл, делает предсказания и отображает результаты.
    """
    try:
        print(f"\nПолучен файл: {file.filename}")
        file_content = await file.read()
        df = process_uploaded_file(file_content)

        # Вывод данных для отладки
        print("\n--- DataFrame перед масштабированием (должен содержать 30 признаков): ---")
        print(df.head())
        print(f"Форма DataFrame перед масштабированием: {df.shape}")

        # Масштабируем данные с помощью загруженного скейлера
        # Важно: используем scaler.transform(), а не fit_transform()
        scaled_data = scaler.transform(df)

        print("\n--- Данные после scaler.transform: ---")
        print(f"Форма масштабированных данных: {scaled_data.shape}")
        print("Первые 5 строк масштабированных данных:")
        print(scaled_data[:5]) # Выводим первые 5 строк масштабированных данных

        # Делаем предсказания
        preds = model.predict(scaled_data)
        # Получаем вероятности для класса 1 (болен)
        probs = model.predict_proba(scaled_data)[:, 1]

        print("\n--- Результаты предсказаний ---")
        print("Предсказания (preds):", preds)
        print("Вероятности класса 1 (probs):", probs.round(3))

        # Добавляем предсказания и вероятности в DataFrame для отображения
        df["Prediction"] = preds
        df["Probability_class_1"] = probs.round(3)

        # Заменяем числовые предсказания на понятные метки
        df["Diagnosis"] = df["Prediction"].apply(lambda x: "Болен" if x == 1 else "Здоров")

        # Преобразуем DataFrame в HTML таблицу
        tables = df.to_html(classes="table table-striped", index=False)

        print("\n--- Предсказания успешно выполнены и готовы к отображению ---")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "tables": tables,
            "success_message": "Предсказания успешно выполнены!"
        })

    except ValueError as ve:
        print(f"Ошибка данных (ValueError): {ve}") # Логирование ошибки
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Ошибка данных: {ve}"
        })
    except Exception as e:
        print(f"Произошла непредвиденная ошибка (Exception): {e}") # Логирование ошибки
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Произошла непредвиденная ошибка: {e}. Пожалуйста, попробуйте еще раз."
        })
       
