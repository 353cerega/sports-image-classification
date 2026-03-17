from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mlflow.tracking import MlflowClient

plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

client = MlflowClient(tracking_uri="file:./mlruns")

experiments = client.search_experiments(order_by=["last_update_time DESC"])
if not experiments:
    print("Не найдено ни одного эксперимента.")
    exit(1)

latest_exp = experiments[0]
exp_id = latest_exp.experiment_id
print(f"Используем эксперимент: {latest_exp.name} (ID: {exp_id})")

runs = client.search_runs(experiment_ids=[exp_id], order_by=["start_time DESC"])
if not runs:
    print("В эксперименте нет запусков.")
    exit(1)

latest_run = runs[0]
run_id = latest_run.info.run_id
print(f"Используем запуск: {run_id}")

metrics = ["train_loss", "val_loss", "train_acc", "val_acc"]
data = {}
for metric in metrics:
    history = client.get_metric_history(run_id, metric)
    if history:
        steps = [item.step for item in history]
        values = [item.value for item in history]
        data[metric] = pd.DataFrame({"step": steps, metric: values})
    else:
        print(f"Метрик {metric} не найдено.")

if not data:
    print("Нет данных для построения графиков.")
    exit(1)

plt.figure(figsize=(10, 5))
if "train_loss" in data:
    plt.plot(
        data["train_loss"]["step"], data["train_loss"]["train_loss"], label="Train Loss"
    )
if "val_loss" in data:
    plt.plot(data["val_loss"]["step"], data["val_loss"]["val_loss"], label="Val Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(plots_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
print("График потерь сохранён в plots/loss_curves.png")

# График точности
plt.figure(figsize=(10, 5))
if "train_acc" in data:
    plt.plot(
        data["train_acc"]["step"],
        data["train_acc"]["train_acc"],
        label="Train Accuracy",
    )
if "val_acc" in data:
    plt.plot(data["val_acc"]["step"], data["val_acc"]["val_acc"], label="Val Accuracy")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(plots_dir / "acc_curves.png", dpi=150, bbox_inches="tight")
print("График точности сохранён в plots/acc_curves.png")
