import argparse
from logging import warning
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

from realm_tune.settings import RealmTuneConfig, HpTuningType
class OptunaHyperparamTuner:
    def __init__(self, config: RealmTuneConfig):
        self.config = config
        pass

    def __call__(self, trial: optuna.trial.Trial) -> float:
        '''
        Optuna's objective function
        '''
        print(f'Running trial {trial.number}')

        run_id = f"{self.config['realm_ai']['behavior_name']}_{trial.number}"
        if self.use_wandb:
            wandb_run = wandb.init(entity=self.config['realm_ai']['wandb']['entity'], group=self.config['realm_ai']['folder_name'], project=self.config['realm_ai']['wandb']['project'], reinit=True, sync_tensorboard=True, job_type=self.wandb_jobtype)
            wandb_run.name = f"{run_id}_{wandb_run.id}"
        
        curr_config = deepcopy(self.config['mlagents'])
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
            tmp_pointer = curr_config['default_settings']
            for i in self.hyperparams_path[hyperparam][:-1]:
                tmp_pointer = tmp_pointer[i]
            tmp_pointer[hyperparam] = val
            
            if self.use_wandb:
                wandb_run.config[hyperparam] = val
        

        self.__create_config_file(run_id, curr_config)

        subprocess.run(["mlagents-learn", f"{run_id}.yml", "--force"])
        # run_cli(parse_command_line(argv=[f"{run_id}.yml", "--force"]))

        # if self.tensorboard_url is None:
        #     self.__launch_tensorboard()
        score = self.__evaluate(run_id)
        print(f'Score for trial {trial.number}: {score}')

        if self.use_wandb:
            wandb.summary["Average Reward"] = score
            # wandb.Api(overrides={'entity':self.config['realm_ai']['wandb']['entity']}).sync_tensorboard(root_dir=f"./results/{run_id}/{self.config['realm_ai']['behavior_name']}", run_id=wandb_run.id, project=self.config['realm_ai']['wandb']['project'])
            wandb_run.finish()

        return score

    def __create_config_file(self, run_id, config):
        '''
        Create a config file for the given configuration
        '''
        config['checkpoint_settings']['run_id'] = run_id
        with open(f'{run_id}.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False) 

    def __evaluate(self, run_id) -> int: 
        logdir = f"./results/{run_id}/{self.config['realm_ai']['behavior_name']}/events*"
        eventfile = glob.glob(logdir)[0]
        rew = [value.simple_value 
        for serialized_example in tf.data.TFRecordDataset(eventfile) 
            for value in event_pb2.Event.FromString(serialized_example.numpy()).summary.value 
                if value.tag == 'Environment/Cumulative Reward']
        return statistics.mean(rew[-self.config['realm_ai']['eval_window_size']:])

    def __get_sampler(self)-> optuna.samplers.BaseSampler:
        if self.algo == 'bayes':
            return TPESampler(n_startup_trials=self.config['realm_ai']['warmup_trials']) # TODO: change to have default values instead when loading config
        elif self.algo == 'random':
            return RandomSampler()
        elif self.algo == 'grid':
            search_space = {i:v for i, _, v in self.hyperparameters_to_tune}
            return GridSampler(search_space)

    