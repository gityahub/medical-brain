import yaml
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os

# =======================
# Dataclass Definitions
# =======================

@dataclass
class SingleDatasetConfig:
    root_dir: str
    modalities: List[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    missing_modal_prob: float

@dataclass
class DataConfig:
    default_dataset: str
    datasets: Dict[str, SingleDatasetConfig]

@dataclass
class ModelConfig:
    encoder_channels: List[int]
    encoder_depth: int
    image_size: Tuple[int, int, int]
    patch_size: Tuple[int, int, int]
    projection_dim: int
    prototype_num: int
    mae_mask_ratio: float
    uncertainty_threshold: float
    contrastive_temperature: float
    dynamic_modal_weight: bool

@dataclass
class TrainingConfig:
    ssl_epochs: int
    ssl_batch_size: int
    ssl_lr: float
    optimizer: str
    weight_decay: float
    sup_epochs: int
    sup_batch_size: int
    sup_lr: float
    distill_weight: float

@dataclass
class LossConfig:
    mae_weight: float
    contrastive_weight: float
    prototype_weight: float
    seg_weight: float
    cls_weight: float
    kd_weight: float

@dataclass
class LoggingConfig:
    save_dir: str
    tensorboard_dir: str
    ckpt_interval: int

@dataclass
class FullConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    logging: LoggingConfig


# =======================
# Load Function
# =======================

def load_config(path: str) -> FullConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    datasets = {
        name: SingleDatasetConfig(**cfg)
        for name, cfg in raw["data"]["datasets"].items()
    }

    data_cfg = DataConfig(
        default_dataset=raw["data"]["default_dataset"],
        datasets=datasets,
    )

    cfg = FullConfig(
        data=data_cfg,
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        loss=LossConfig(**raw["loss"]),
        logging=LoggingConfig(**raw["logging"]),
    )

    validate_config(cfg)
    return cfg


# =======================
# Validation
# =======================

def validate_config(cfg: FullConfig):
    for img, patch in zip(cfg.model.image_size, cfg.model.patch_size):
        assert img % patch == 0, f"patch_size {cfg.model.patch_size} can't be divisible by image_size {cfg.model.image_size}"

    assert cfg.data.default_dataset in cfg.data.datasets, \
        f"default_dataset {cfg.data.default_dataset} not exists in datasets"

    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    os.makedirs(cfg.logging.tensorboard_dir, exist_ok=True)
    print(f"Config loaded: {cfg.data.default_dataset}")

# =======================
# Example Usage
# =======================
if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    ds_name = cfg.data.default_dataset
    ds_cfg = cfg.data.datasets[ds_name]
    print(f"Using dataset: {ds_name}")
    print(f"Root: {ds_cfg.root_dir}")
    print(f"Modalities: {ds_cfg.modalities}")
