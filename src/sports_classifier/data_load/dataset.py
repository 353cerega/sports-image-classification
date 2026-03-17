from pathlib import Path

import kaggle
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def download_data(data_path: str = None) -> None:
    if data_path is None:
        project_root = Path(__file__).parents[3]
        data_path = project_root / "data"
    else:
        data_path = Path(data_path)

    data_path.mkdir(parents=True, exist_ok=True)

    dataset_folder = data_path / "sports-classification"
    if dataset_folder.exists():
        print(f"Dataset already exists in {dataset_folder}. Skipping download.")
        return

    print(f"Downloading dataset to {data_path}...")
    kaggle.api.dataset_download_files(
        "gpiosenka/sports-classification", path=str(data_path), unzip=True, quiet=False
    )
    print("Download complete.")


class SportsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=32, num_workers=4, image_size=224):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.test_transform = self.val_transform

    def prepare_data(self):
        if not self.data_dir.exists() or not any(self.data_dir.iterdir()):
            download_data(str(self.data_dir))

    def setup(self, stage=None):
        train_path = self.data_dir / "train"
        val_path = self.data_dir / "valid"
        test_path = self.data_dir / "test"

        for p in [train_path, val_path, test_path]:
            if not p.exists():
                raise FileNotFoundError(f"Папка {p} не найдена.")

        self.train_dataset = ImageFolder(
            root=train_path, transform=self.train_transform
        )
        self.val_dataset = ImageFolder(root=val_path, transform=self.val_transform)
        self.test_dataset = ImageFolder(root=test_path, transform=self.test_transform)

        self.classes = self.train_dataset.classes
        self.num_classes = len(self.classes)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
