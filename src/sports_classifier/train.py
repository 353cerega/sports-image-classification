import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from src.sports_classifier.data_load.dataset import SportsDataModule
from src.sports_classifier.models.model import SportsClassifier


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):

    dm = SportsDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=cfg.data.image_size,
    )
    dm.prepare_data()
    dm.setup()

    model = SportsClassifier(
        model_name=cfg.model.model_name,
        num_classes=dm.num_classes,
        learning_rate=cfg.model.learning_rate,
    )

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
    )
    mlf_logger.log_hyperparams(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_acc:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=mlf_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        deterministic=True,
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
