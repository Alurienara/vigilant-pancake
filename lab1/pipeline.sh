set -e
echo "Устанавливаем зависимости..."
pip install --user numpy pandas scikit-learn
echo "=== Шаг 1: Генерация данных ==="
python3 data_creation.py
echo "=== Шаг 2: Предобработка данных ==="
python3 data_preprocessing.py
echo "=== Шаг 3: Обучение модели ==="
python3 model_preparation.py
echo "=== Шаг 4: Тестирование модели ==="
python3 model_testing.py

echo "Пайплайн завершён."