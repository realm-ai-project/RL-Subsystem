
from typing import Any, Mapping


from realm_tune.settings import WandBSettings

def add_wandb_config(config_dict: Mapping[str, Any], wandb_settings:WandBSettings) -> None:
    config_dict['wandb'] = wandb_settings.to_dict()