from typing import Any, Mapping

from mlagents_envs.environment import UnityEnvironment
import yaml

from realm_tune.settings import WandBSettings

def add_wandb_config(config_dict: Mapping[str, Any], wandb_settings:WandBSettings) -> None:
    config_dict['wandb'] = wandb_settings.to_dict()

def assert_singleplayer_env(env_path):
    _env = UnityEnvironment(env_path, no_graphics=True)
    if not _env.behavior_specs:
        _env.step()
    try:
        if len(_env.behavior_specs) != 1:
            raise Exception(
                "Realm Tune only works with single player environments for now"
            )
    finally:
        _env.close()

def load_yaml_file(path_to_yaml_file): 
    with open(path_to_yaml_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config