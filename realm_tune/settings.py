import argparse
from typing import Dict, Optional, Set, Union
import warnings

import attr
import cattr
from cattr.converters import Converter
import wandb
import yaml

from realm_tune.cli_config import DetectDefault, parser

@attr.s(auto_attribs=True)
class WandBSettings:
    wandb: bool = False        
    wandb_project: Union[str, None] = parser.get_default('wandb_project')
    wandb_entity: Union[str, None] = parser.get_default('wandb_entity')
    wandb_offline: bool = False
    wandb_group: Union[str, None] = parser.get_default('wandb_group')
    wandb_jobtype: Union[str, None] = parser.get_default('wandb_jobtype')
    
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
    env_path: Union[str, None] = parser.get_default('env_path')
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

@attr.s(auto_attribs=True)
class RealmTuneConfig:
    realm_ai: RealmTuneBaseConfig = attr.ib(factory=RealmTuneBaseConfig)
    mlagents: Dict = attr.Factory(dict)

    @staticmethod
    def from_yaml_file(path_to_yaml_file): 
        converter = Converter()
        converter.register_structure_hook(WandBSettings, WandBSettings.structure_from_yaml)
        with open(path_to_yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Ensure that all fields in realm_ai portion of yaml file are valid - we leave validation of mlagents and values of the fields to their own functions
        if 'realm_ai' in config:
            for k, v in config['realm_ai'].items():
                assert k in attr.fields_dict(RealmTuneBaseConfig), f'"{k}" field is not supported in {path_to_yaml_file}!'
        return converter.structure(config, RealmTuneConfig)

    @staticmethod
    def from_argparse(parser: argparse.ArgumentParser):
        args = parser.parse_args()
        realm_tune_config = RealmTuneConfig.from_yaml_file(args.config_path)
        dict_ = cattr.unstructure(realm_tune_config)
        for k, v in vars(args).items():
            if k in DetectDefault.non_default_args:
                if k in attr.fields_dict(RealmTuneBaseConfig):
                    dict_['realm_ai'][k] = v
                elif k in attr.fields_dict(WandBSettings):
                    dict_['realm_ai']['wandb'][k] = v
                else:
                    if k != "config_path": warnings.warn(f'"{k}" field in yaml file not supported, and will be ignored')
        return cattr.structure(dict_, RealmTuneConfig)

# TODO: Look into "if key in DetectDefault.non_default_args:" line 878, settings.py
# The idea is that we don't want to blindly update everything based on our argparse, we want to only update our settings based on argparse if it has been changed from its default value!

# Solution: First, update all values using those from the config file. Then, Add a custom structure hook for each type. In the hook, iterate through argparse, and check if argument has been changed from default value. If yes, change value in dictionary. Else, don't add key to dictionary. Then just do cattr.structure(...) 

if __name__=='__main__':
    # args = parser.parse_args(["--config-path","realm_tune/bayes.yaml"])
    # item = RealmTuneConfig.from_yaml_file(args.config_path)
    # print(item)

    print(RealmTuneConfig.from_argparse(parser))