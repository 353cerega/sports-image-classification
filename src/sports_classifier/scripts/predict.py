import requests
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from src.sports_classifier.data.dataset import SportsDataModule


def load_class_names(data_dir="data"):
    dm = SportsDataModule(data_dir=data_dir)
    dm.prepare_data()
    dm.setup()
    return dm.classes


def preprocess_image(image_path, image_size=224):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).numpy()


def predict_image(image_path, server_url="http://127.0.0.1:5001/invocations", data_dir="data"):
    """Отправляет изображение на сервер и выводит предсказание."""
    try:
        class_names = load_class_names(data_dir)
        print(f"Загружено {len(class_names)} классов")
    except Exception as e:
        print(f"Не удалось загрузить классы: {e}")
        class_names = [str(i) for i in range(100)]

    try:
        input_tensor = preprocess_image(image_path)
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return

    payload = {"inputs": input_tensor.tolist()}
    print(f"Отправка запроса на {server_url}...")

    try:
        response = requests.post(server_url, json=payload)
    except requests.exceptions.ConnectionError:
        print("Ошибка подключения к серверу. Убедитесь, что MLflow сервер запущен (poetry run mlflow models serve -m mlflow_model -p 5001 --no-conda)")
        return

    if response.status_code != 200:
        print(f"Ошибка сервера: {response.status_code} - {response.text}")
        return

    result = response.json()
    if isinstance(result, dict) and 'predictions' in result:
        predictions = result['predictions'][0]
    elif isinstance(result, list):
        predictions = result[0]
    else:
        predictions = result

    exp_preds = np.exp(predictions - np.max(predictions))
    probabilities = exp_preds / np.sum(exp_preds)

    predicted_idx = int(np.argmax(predictions))
    confidence = float(np.max(probabilities))

    predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else f"класс {predicted_idx}"
    print(f"\nПредсказанный класс: {predicted_class}")
    print(f"Уверенность: {confidence:.4f}")

    top5_idx = np.argsort(predictions)[-5:][::-1]
    print("\nТоп-5 предсказаний:")
    for i, idx in enumerate(top5_idx):
        class_name = class_names[idx] if idx < len(class_names) else f"класс {idx}"
        print(f"  {i+1}. {class_name} (вероятность: {probabilities[idx]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sport from image using MLflow server")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--port", type=int, default=5001, help="MLflow server port")
    parser.add_argument("--host", default="127.0.0.1", help="MLflow server host")
    parser.add_argument("--data_dir", default="data", help="Data directory (to load class names)")
    args = parser.parse_args()

    server_url = f"http://{args.host}:{args.port}/invocations"
    predict_image(args.image_path, server_url, args.data_dir)
