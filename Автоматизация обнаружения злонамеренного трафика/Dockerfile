# Основной образ - Python 3.9
FROM python:3.9-slim

# Устанавливаем рабочий каталог
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .

# Устанавливаем необходимые пакеты
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Копируем модель и код приложения в контейнер
COPY ["lightgbm_best_model.txt", "main.py", "./"] .

# Запускаем приложение с помощью gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8989", "main:app"]