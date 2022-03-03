import argparse
from logging import warning
import warnings
import cattr
import yaml
from copy import deepcopy
import os
import subprocess 
import time
import statistics
import pickle
import shutil
import glob

# import requests
# from tensorboard import program
import tensorflow as tf
from tensorflow.core.util import event_pb2
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
import wandb
from mlagents.trainers.learn import run_cli, parse_command_line
from mlagents_envs.environment import UnityEnvironment # We can extract behavior-name from here!

from realm_tune.settings import MLAgentsBaseConfig, RealmTuneConfig, HpTuningType
from realm_tune.utils import add_wandb_config

class OptunaHyperparamTuner:

    def __init__(self, options: RealmTuneConfig, config_save_path:str = "./trial_config/"):
        self.options: RealmTuneConfig = options
        self.hyperparameters_to_tune = self.options.mlagents.hyperparameters_to_tune
        self.hyperparams_path = self.options.mlagents.hyperparams_path
        self.config_save_path = config_save_path
        os.makedirs(self.config_save_path, exist_ok=True)

    def __call__(self, trial: optuna.trial.Trial) -> float:
        '''
        Optuna's objective function
        '''
        print(f'Running trial {trial.number}')

        config_filename = f"{self.options.realm_ai.behavior_name}_{trial.number}"
        run_id = f"{config_filename}_run"

        curr_config = deepcopy(self.options.mlagents)
        for hyperparam, hpTuningType, values in self.hyperparameters_to_tune:
            if hpTuningType==HpTuningType.CATEGORICAL:
                val = trial.suggest_categorical(hyperparam, values)
            elif hpTuningType==HpTuningType.UNIFORM_FLOAT:
                val = trial.suggest_float(hyperparam, values[0], values[1])
            elif hpTuningType==HpTuningType.LOG_UNIFORM_FLOAT:
                val = trial.suggest_float(hyperparam, values[0], values[1], log=True)
            elif hpTuningType==HpTuningType.UNIFORM_INT:
                val = trial.suggest_int(hyperparam, values[0], values[1])
            elif hpTuningType==HpTuningType.LOG_UNIFORM_INT:
                val = trial.suggest_int(hyperparam, values[0], values[1], log=True)
            else:
                raise NotImplementedError(f'{hpTuningType}: Unknown type of hyperparameter')
            
            # Traverse recursively into config dictionary to replace value
            tmp_pointer = curr_config.default_settings
            for i in self.hyperparams_path[hyperparam][:-1]:
                tmp_pointer = tmp_pointer[i]
            tmp_pointer[hyperparam] = val
        
        file_dir = self._create_config_file(run_id, config_filename, curr_config)

        p = subprocess.Popen(["wandb-mlagents-learn", file_dir, "--force"])
        p.wait()

        score = self._evaluate(run_id)
        print(f'Score for trial {trial.number}: {score}')

        return score

    def _create_config_file(self, run_id: str, config_filename: str, config: MLAgentsBaseConfig):
        '''
        Create a config file for the given configuration
        '''
        config.checkpoint_settings['run_id'] = run_id
        config_dict = config.to_dict()
        if self.options.realm_ai.wandb.use_wandb:
            add_wandb_config(config_dict, self.options.realm_ai.wandb)
        file_dir = os.path.join(self.config_save_path, f'{config_filename}.yml')
        with open(file_dir, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False) 
        return file_dir

    def _evaluate(self, run_id: str) -> int: 
        logdir = f"./results/{run_id}/*/events.out.tfevents*"
        eventfiles = glob.glob(logdir)
        assert len(eventfiles)>0, "TensorBoard event file not found!"
        if len(eventfiles)>1:
            warnings.warn("Multiple TensorBoard event files found, using the first one...")
        eventfile = eventfiles[0]
        rew = [value.simple_value 
        for serialized_example in tf.data.TFRecordDataset(eventfile) 
            for value in event_pb2.Event.FromString(serialized_example.numpy()).summary.value 
                if value.tag == 'Environment/Cumulative Reward']
        return statistics.mean(rew[-self.options.realm_ai.eval_window_size:])


    