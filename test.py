from config.model_config import load_config

cfg = load_config("config/config.yaml")
ds_name = cfg.data.default_dataset
ds_cfg = cfg.data.datasets[ds_name]
print(f"Using dataset: {ds_name}")
print(f"Root: {ds_cfg.root_dir}")
print(f"Modalities: {ds_cfg.modalities}")