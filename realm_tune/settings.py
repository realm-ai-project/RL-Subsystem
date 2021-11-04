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
                if k=='project' or k=='wandb_project':
                    dict_['wandb_project'] = v
                elif k=='entity' or k=='wandb_entity':
                    dict_['wandb_entity'] = v
                elif k=='group' or k=='wandb_group':
                    dict_['wandb_group'] = v
                elif k=='offline' or k=='wandb_offline':
                    dict_['wandb_offline'] = v
                elif k=='jobtype'or k=="wandb_jobtype":
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
class MLAgentsBaseConfig: 
    '''
    Leave these as dictionaries, validation of the fields will be performed when mlagents is called.
    '''
    # Don't strictly need this, just here for backup
    default_settings: Dict = attr.ib(factory=lambda :{'default_settings': {'trainer_type': 'ppo', 'hyperparameters': {'batch_size': 'log_unif(64, 16384)', 'buffer_size': 'log_unif(2000, 50000)', 'learning_rate': 'log_unif(0.0003, 0.01)', 'beta': 'log_unif(0.001, 0.03)', 'epsilon': 0.2, 'lambd': 'unif(0.95, 1.0)', 'num_epoch': 3, 'learning_rate_schedule': 'linear'}, 'network_settings': {'normalize': True, 'hidden_units': [64, 256, 512, 1024], 'num_layers': 'unif(1, 3)', 'vis_encode_type': 'simple'}, 'reward_signals': {'extrinsic': {'gamma': 'unif(0.9, 1.0)', 'strength': 1.0}}, 'keep_checkpoints': 5, 'max_steps': 100000, 'time_horizon': 1000, 'summary_freq': 10000}})
    env_settings: Dict = attr.ib(factory=dict)
    engine_settings: Dict = attr.ib(factory=dict)
    environment_parameters: Dict = attr.ib(factory=dict)
    checkpoint_settings: Dict = attr.ib(factory=dict)
    torch_settings: Dict = attr.ib(factory=dict)
    debug: bool = False


@attr.s(auto_attribs=True)
class RealmTuneConfig:
    realm_ai: RealmTuneBaseConfig = attr.ib(factory=RealmTuneBaseConfig)
    mlagents: MLAgentsBaseConfig = attr.ib(factory=MLAgentsBaseConfig)

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


# For debugging purposes
if __name__=='__main__':
    # args = parser.parse_args(["--config-path","realm_tune/bayes.yaml"])
    args = parser.parse_args(["--config-path","test.yml"])
    item = RealmTuneConfig.from_yaml_file(args.config_path)
    print(item)

    config = cattr.unstructure(RealmTuneConfig.from_argparse(parser))
    with open(f'test.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)