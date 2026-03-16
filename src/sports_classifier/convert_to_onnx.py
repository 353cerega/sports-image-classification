import torch
import torch.onnx
from src.sports_classifier.models.model import SportsClassifier

def convert_to_onnx(checkpoint_path: str, output_path: str = "models/model.onnx"):
    model = SportsClassifier(model_name="mobilenet_v3_small", num_classes=100)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print(f"Модель сохранена в {output_path}")

if __name__ == "__main__":
    convert_to_onnx("mlruns/156929407276084794/c57139e795474f57bb8b4b147a6e2444/checkpoints/best-epoch=01-val_acc=0.7840.ckpt")
