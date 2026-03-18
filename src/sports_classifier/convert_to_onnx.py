import argparse
from pathlib import Path

import torch
import torch.onnx

from src.sports_classifier.models.model import SportsClassifier


def find_latest_checkpoint():
    """Ищет самый свежий .ckpt файл во всём проекте (рекурсивно)."""
    ckpt_files = list(Path.cwd().rglob("*.ckpt"))
    if not ckpt_files:
        return None
    # Самый свежий по времени модификации
    latest = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    return latest


def convert_to_onnx(checkpoint_path: Path, output_dir: Path = Path("models")):
    """
    Загружает модель из чекпоинта и экспортирует в ONNX.
    """
    print(f"Загрузка чекпоинта: {checkpoint_path}")

    export_device = torch.device("cpu")
    model = SportsClassifier.load_from_checkpoint(
        checkpoint_path, map_location=export_device
    )
    model.to(export_device)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"

    dummy_input = torch.randn(1, 3, 224, 224, device=export_device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Модель успешно сохранена в {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert trained model to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint file (optional)"
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

    convert_to_onnx(checkpoint_path)


if __name__ == "__main__":
    main()
