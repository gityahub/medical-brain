from config.model_config import Config
config = Config.from_yaml('config/config.yaml')
print(config.encoder.base_channels)