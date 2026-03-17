import argparse
import shutil
from pathlib import Path

import mlflow
import mlflow.pyfunc
import torch

from src.sports_classifier.models.model import SportsClassifier


def find_latest_checkpoint():
    """Ищет самый свежий .ckpt файл во всём проекте (рекурсивно)."""
    ckpt_files = list(Path.cwd().rglob("*.ckpt"))
    if not ckpt_files:
        return None
    # Самый свежий по времени модификации
    latest = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    return latest


def save_mlflow_model(
    checkpoint_path: Path, output_dir: Path = Path("mlflow_model"), force: bool = False
):
    """
    Загружает модель из чекпоинта и сохраняет в формате MLflow.
    """
    # Проверка существования выходной папки
    if output_dir.exists() and any(output_dir.iterdir()):
        if force:
            print(f"Папка {output_dir} уже существует и не пуста. Удаляем (--force).")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Папка {output_dir} уже существует. \
            Используйте --force для перезаписи или удалите вручную."
            )

    print(f"Загрузка чекпоинта: {checkpoint_path}")

    model = SportsClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)

    class ModelWrapper(mlflow.pyfunc.PythonModel):
        def __init__(self, model):
            self.model = model

        def predict(self, context, input_data):
            import numpy as np

            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.tensor(input_data, dtype=torch.float32)
                else:
                    input_tensor = torch.tensor(input_data, dtype=torch.float32)
                outputs = self.model(input_tensor)
                return outputs.numpy()

    mlflow.pyfunc.save_model(
        path=str(output_dir),
        python_model=ModelWrapper(model),
        artifacts={},
        conda_env=None,
    )
    print(f"Модель успешно сохранена в {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Save trained model in MLflow format")
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint file (optional)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing mlflow_model directory"
    )
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Ошибка: файл {checkpoint_path} не найден")
            return
    else:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("Не найден ни один .ckpt файл. Укажите путь через --checkpoint")
            return
        print(f"Используется последний чекпоинт: {checkpoint_path}")

    save_mlflow_model(checkpoint_path, force=args.force)


if __name__ == "__main__":
    main()
