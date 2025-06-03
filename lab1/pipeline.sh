# Включаем «строгий» режим (exit on error)
set -e

#  --- 1) Установка зависимостей (если нужно) ---
# Проверим, есть ли virtualenv, иначе просто ставим через pip
# Предполагаем, что pip и python3 уже есть в системе. 
# Если вы хотите изолировать окружение, можно предварительно сделать: python3 -m venv venv && source venv/bin/activate

echo "Устанавливаем зависимости..."
pip install --user numpy pandas scikit-learn

#  --- 2) Запуск генерации данных ---
echo "=== Шаг 1: Генерация данных ==="
python3 data_creation.py

#  --- 3) Предобработка данных ---
echo "=== Шаг 2: Предобработка данных ==="
python3 data_preprocessing.py

#  --- 4) Обучение модели ---
echo "=== Шаг 3: Обучение модели ==="
python3 model_preparation.py

#  --- 5) Тестирование модели ---
echo "=== Шаг 4: Тестирование модели ==="
python3 model_testing.py

echo "Пайплайн завершён."
