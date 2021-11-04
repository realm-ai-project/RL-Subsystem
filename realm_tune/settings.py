import argparse
from typing import Dict, Optional, Set, Union
from enum import Enum, auto
import warnings
import os

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

    # def __attrs_post_init__(self): # No need, done in wandb wrapper
    #     if self.wandb_offline:
    #         os.environ["WANDB_MODE"]="offline"
    
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
    behavior_name: Optional[str] = parser.get_default('behavior_name')
    algorithm: str = attr.ib(default=parser.get_default('algorithm'))
    @algorithm.validator
    def _check_algorithm(self, attribute, value):
        assert value in ['bayes', 'grid', 'random']
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

class HpTuningType(Enum):
    CATEGORICAL = auto()
    UNIFORM_INT = auto()
    UNIFORM_FLOAT = auto()
    LOG_UNIFORM_FLOAT = auto()
    LOG_UNIFORM_INT = auto()

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

    def find_hyperparameters_to_tune(self, algo):
        '''
        Find all hyperparameters to perform hyperparameter tuning on, 
        and store them in self.hyperparams_path and self.hyperparameters_to_tune
        '''
        hyperparameters = self.default_settings
        # tells us path to hyperparameter in config file given hyperparameter name
        self.hyperparams_path = {}
        # a tuple of three values (hyperparameter, HpTuningType, tuple of values)
        self.hyperparameters_to_tune = []

        def parse_recursively(dict_, path: list):
            if not dict_:
                return 
            for k, v in dict_.items():
                # if its a list, its a categorical hyperparameter to tune
                if isinstance(v, list):
                    self.hyperparameters_to_tune.append((k, HpTuningType.CATEGORICAL, tuple(v)))
                    if k in self.hyperparams_path:
                        raise Exception(f'Duplicated hyperparameters found!, hyperparameter: {k}')
                    self.hyperparams_path[k] = path+[k]
                # if it contains the substring "unif(" and ")" or "log_unif(" and ")", its a continuous hyperparameter to tune
                elif isinstance(v, str) and '(' in v and ')' in v:
                    assert algo!='grid', f'Continuous hyperparameters "{k}" not allowed in grid search!'
                    assert v[-1]==')', 'Continuous hyperparameter must end in ")"'
                    index = v.find('(')
                    list_ = v[index+1:-1].split(',')
                    assert(len(list_)==2), f'Parsing "{v}" for hyperparameter "{k}" failed; Continuous hyperparameter(s) must only contain 2 numerical values!'
                    low, high = float(list_[0]), float(list_[1])
                    assert(low<high), f'Parsing "{v}" for hyperparameter "{k}" failed; First value must be smaller than the second!'
                    
                    if v[:index]=='unif':
                        if low.is_integer() and high.is_integer():
                            self.hyperparameters_to_tune.append((k, HpTuningType.UNIFORM_INT, (int(low), int(high))))
                        else:
                            self.hyperparameters_to_tune.append((k, HpTuningType.UNIFORM_FLOAT, (low, high)))
                    elif v[:index]=='log_unif':
                        if low.is_integer() and high.is_integer():
                            self.hyperparameters_to_tune.append((k, HpTuningType.LOG_UNIFORM_INT, (int(low), int(high))))
                        else:
                            self.hyperparameters_to_tune.append((k, HpTuningType.LOG_UNIFORM_FLOAT, (low, high)))
                    else:
                        raise NotImplementedError(f'"{v[:index]}" method not implemented!')
                    self.hyperparams_path[k] = path+[k]
                # recursively parse nested dictionary
                elif isinstance(v, dict):
                    parse_recursively(dict_[k], path+[k])

        parse_recursively(hyperparameters, list())


@attr.s(auto_attribs=True)
class RealmTuneConfig:
    realm_ai: RealmTuneBaseConfig = attr.ib(factory=RealmTuneBaseConfig)
    mlagents: MLAgentsBaseConfig = attr.ib(factory=MLAgentsBaseConfig)

    def validate(self):
        # Ensure that env_path is set somewhere
        if self.realm_ai.env_path is None and 'env_path' not in self.mlagents.env_settings:
            raise ValueError('Realm-tune does not support in-editor training! Please pass in a --config-path flag, or add env_path to yaml file under the mlagents config')
        if 'env_path' not in self.mlagents.env_settings:
            self.mlagents.env_settings['env_path'] = self.realm_ai.env_path
        
        # Find hyperparameters to tune
        self.mlagents.find_hyperparameters_to_tune(self.realm_ai.algorithm)

        # TODO: Find behavior name if it is not passed in
        # For now, assert that behavior name is passed in
        assert self.realm_ai.behavior_name is not None, "We need a behavior name!"


    @staticmethod
    def _from_yaml_file(path_to_yaml_file): 
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
    def from_argparse(args: argparse.Namespace):
        realm_tune_config = RealmTuneConfig._from_yaml_file(args.config_path)
        dict_ = cattr.unstructure(realm_tune_config)
        for k, v in vars(args).items():
            if k in DetectDefault.non_default_args:
                if k in attr.fields_dict(RealmTuneBaseConfig):
                    dict_['realm_ai'][k] = v
                elif k in attr.fields_dict(WandBSettings):
                    dict_['realm_ai']['wandb'][k] = v
                else:
                    if k != "config_path": warnings.warn(f'"{k}" field in yaml file not supported, and will be ignored')
        realm_tune_config = cattr.structure(dict_, RealmTuneConfig)
        realm_tune_config.validate()
        return realm_tune_config


# For debugging purposes
if __name__=='__main__':
    # args = parser.parse_args(["--config-path","realm_tune/bayes.yaml"])
    # args = parser.parse_args(["--config-path","test.yml"])
    # item = RealmTuneConfig.from_yaml_file(args.config_path)
    # print(item)

    config = cattr.unstructure(RealmTuneConfig.from_argparse(parser.parse_args()))
    print(config)
    # with open(f'test.yaml', 'w') as f:
            # yaml.dump(config, f, default_flow_style=False)