import mlflow
import mlflow.pyfunc
import torch
from src.sports_classifier.models.model import SportsClassifier

CHECKPOINT_PATH = "mlruns/156929407276084794/246388036b13480c860c06cdc71e929b/checkpoints/best-epoch=01-val_acc=0.4320.ckpt"
MODEL_OUTPUT_DIR = "mlflow_model"

model = SportsClassifier(
    model_name="mobilenet_v3_small",
    num_classes=100,
    learning_rate=0.001
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, input_data):
        import numpy as np
        import torch
        from torchvision import transforms

        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                input_tensor = input_data
            outputs = self.model(input_tensor)
            return outputs.numpy()

mlflow.pyfunc.save_model(
    path=MODEL_OUTPUT_DIR,
    python_model=ModelWrapper(model),
    artifacts={},
    conda_env=None,
)

print(f"Модель сохранена в {MODEL_OUTPUT_DIR}")
