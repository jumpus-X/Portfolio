import lightgbm as lgb
import pandas as pd

from flask import Flask, jsonify, request

# Загрузка модели
model = lgb.Booster(model_file="lightgbm_best_model.txt")

# Инициализация приложения
app = Flask("default")

# Настройка конечной точки для предсказания
@app.route("/predict", methods=["POST"])
def predict():
    # Получение предоставленного JSON
    X = request.get_json()
    # Выполнение предсказания
    preds = model.predict(pd.DataFrame(X, index=[0]))
    # Вывод JSON с вероятностями для каждого класса
    result = {"class_probabilities": preds.tolist()}
    return jsonify(result)

if __name__ == "__main__":
    # Запуск приложения на локальном хосте и порту 8989
    app.run(debug=True, host="0.0.0.0", port=8989)
    