from typing import Dict, Optional, Set, Union
import warnings

import attr
import cattr
from cattr.converters import Converter
import wandb
import yaml
from mlagents.trainers.settings import EnvironmentSettings, EngineSettings, EnvironmentParameterSettings, CheckpointSettings, TorchSettings

from realm_tune.cli_config import parser

@attr.s(auto_attribs=True)
class WandBSettings:
    wandb: bool = False        
    wandb_project: Union[str, None] = parser.get_default('wandb_project')
    wandb_entity: Union[str, None] = parser.get_default('wandb_entity')
    wandb_offline: bool = False
    wandb_group: Union[str, None] = parser.get_default('wandb_group')
    wandb_jobtype: Union[str, None] = parser.get_default('wandb_jobtype')

    # @staticmethod
    # def structure_from_argparse(obj: Dict, type):
    #     dict_ = {}
    #     for k, v in obj.items():
    #         if k=='wandb_project':
    #             dict_['project'] = v
    #         elif k=='wandb_entity':
    #             dict_['entity'] = v
    #         elif k=='wandb_group':
    #             dict_['group'] = v
    #         elif k=='wandb_jobtype':
    #             dict_['jobtype'] = v
    #         elif k=='wandb_offline':
    #             dict_['offline'] = v
    #         elif k=='wandb':
    #             dict_['wandb'] = v
    #     assert dict_['wandb'] == True
    #     return WandBSettings(**dict_)
    
    @staticmethod
    def structure_from_yaml(obj: Dict, _): # Will only enter this function if wandb field exists in yaml file
        dict_ = {'wandb':True}
        if obj is not None: # Possible that their yaml file only has a "wandb:" field
            for k, v in obj.items():
                if k=='project':
                    dict_['wandb_project'] = v
                elif k=='entity':
                    dict_['wandb_entity'] = v
                elif k=='group':
                    dict_['wandb_group'] = v
                elif k=='offline':
                    dict_['wandb_offline'] = v
                elif k=='jobtype':
                    dict_['wandb_jobtype'] = v
                else:
                    warnings.warn(f'"{k}" field in yaml file not supported, and will be ignored')
        return WandBSettings(**dict_)


@attr.s(auto_attribs=True)
class RealmTuneBaseConfig:
    behavior_name: str = parser.get_default('behavior_name')
    algorithm: str = parser.get_default('algorithm')
    total_trials: int = parser.get_default('total_trials')
    warmup_trials: int = parser.get_default('warmup_trials')
    eval_window_size: int = parser.get_default('eval_window_size')
    data_path: Union[str, None] = parser.get_default('data_path')
    wandb: WandBSettings = attr.ib(factory=WandBSettings)

    @staticmethod
    def structure():
        pass
    # @behavior_name.default
    # def _set_default_behavior_name(self):
    #     if parser.get_default('behavior_name') is None:
    #         return 

    # @data_path.default
    # def _set_default_data_path(self):
    #     if parser.get_default('data_path') is None:
    #         return 

# class MLAgentsBaseConfig:
#     default_settings: Dict = attr.Factory(dict)

#     # From ML-Agents
#     env_settings: EnvironmentSettings = attr.ib(factory=EnvironmentSettings)
#     engine_settings: EngineSettings = attr.ib(factory=EngineSettings)
#     environment_parameters: Optional[Dict[str, EnvironmentParameterSettings]] = None
#     checkpoint_settings: CheckpointSettings = attr.ib(factory=CheckpointSettings)
#     torch_settings: TorchSettings = attr.ib(factory=TorchSettings)
#     debug: bool = False


@attr.s(auto_attribs=True)
class RealmTuneConfig:
    realm_ai: RealmTuneBaseConfig = attr.ib(factory=RealmTuneBaseConfig)
    mlagents: Dict = attr.Factory(dict)

    # cattr.register_structure_hook(RealmTuneBaseConfig, RealmTuneBaseConfig.structure)

    @staticmethod
    def from_yaml_file(path_to_yaml_file): 
        converter = Converter()
        converter.register_structure_hook(WandBSettings, WandBSettings.structure_from_yaml)
        with open(path_to_yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # TODO: Ensure that all fields in realm_ai portion of yaml file are valid - we leave validation of mlagents and values of the fields to their own functions
        return converter.structure(config, RealmTuneConfig)

# TODO: Look into "if key in DetectDefault.non_default_args:" line 878, settings.py
# The idea is that we don't want to blindly update everything based on our argparse, we want to only update our settings based on argparse if it has been changed from its default value!

# Solution: First, update all values using those from the config file. Then, Add a custom structure hook for each type. In the hook, iterate through argparse, and check if argument has been changed from default value. If yes, change value in dictionary. Else, don't add key to dictionary. Then just do cattr.structure(...) 

if __name__=='__main__':
    args = parser.parse_args(["--config-path","realm_tune/bayes.yaml"])
    item = RealmTuneConfig.from_yaml_file(args.config_path)
    print(item)