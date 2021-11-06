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

    def run_from_scratch(self, folder_name=None) -> optuna.Study:
        self.folder = folder_name
        if self.folder is None:
            self.folder = f'{self.config["realm_ai"]["behavior_name"]}_{time.strftime("%d-%m-%Y_%H-%M-%S")}'
        
        # if self.use_wandb: # Delete existing runs under the same folder (group) on wandb if it already exists
        #     path = f"{self.config['realm_ai']['wandb']['entity']}/{self.config['realm_ai']['wandb']['project']}"
        #     runs_to_delete = wandb.Api().runs(path=path, filters={"group":self.folder}, per_page=1000)
            
        #     for run in runs_to_delete:
        #         run.delete()

        if os.path.isdir(self.folder):
            raise FileExistsError(f'"{self.folder}/" already exists, and no checkpoint exists in that folder. Please rename folder_name in config file, or delete the folder!')
        else:
            print('Creating folder named', self.folder)
            os.mkdir(self.folder)

        os.chdir(self.folder)

        if self.use_wandb:
            wandb_metadata = {"job_type":time.strftime("%d-%m-%Y_%H-%M-%S")}
            self.wandb_jobtype = wandb_metadata['job_type']
            pickle.dump(wandb_metadata, open( "wandb_metadata.pkl", "wb" ) )

        sampler = self.__get_sampler()
        study = optuna.create_study(study_name=self.folder, sampler=sampler, direction="maximize")
        return study

    def restore_from_checkpoint(self, folder_name) -> optuna.Study:
        self.folder = folder_name
        assert(os.path.isdir(self.folder))

        os.chdir(self.folder)

        sampler = self.__get_sampler()
        new_study = optuna.create_study(study_name=self.folder, sampler=sampler, direction="maximize")
        study = pickle.load( open( "optuna_study.pkl", "rb" ) )
        for trial in study.trials:
            if trial.state.is_finished():
                new_study.add_trial(trial)
        
        if self.use_wandb:
            if not os.path.isfile("wandb_metadata.pkl"):
                warning("Restoring from checkpoint but wandb metadata not found!")
                self.wandb_jobtype = time.strftime("%d-%m-%Y_%H-%M-%S")
            else:
                self.wandb_jobtype = pickle.load( open( "wandb_metadata.pkl", "rb" ) )['job_type']

        # df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        if self.config['realm_ai']['total_trials'] <= len(new_study.trials):
            warning(f'{len(new_study.trials)} completed trials already found in folder "{folder_name}". Exiting...')
            exit(0)
        print(f'Resuming from {len(new_study.trials)} completed trials')
        return new_study

    def run(self):
        '''
        1) Create and cd into new/existing folder 
        2) Create a new Optuna study using the TPESampler, or restore previously saved study
        '''
        if not os.path.isdir('runs'):
            os.mkdir('runs')
        os.chdir('./runs')
        if 'folder_name' in self.config['realm_ai'] and os.path.isdir(self.config['realm_ai']['folder_name']) and os.path.isfile(f"{self.config['realm_ai']['folder_name']}/optuna_study.pkl"):
            study = self.restore_from_checkpoint(self.config['realm_ai']['folder_name'])
        else:
            study = self.run_from_scratch(folder_name=self.config['realm_ai']['folder_name'] if 'folder_name' in self.config['realm_ai'] else None)
        
        interrupted = False
        try:
            study.optimize(self, n_trials=self.config['realm_ai']['total_trials']-len(study.trials))
        except KeyboardInterrupt:
            interrupted = True

        pickle.dump(study, open( "optuna_study.pkl", "wb" ) )
        print('Saved study as optuna_study.pkl')

        print("Number of finished trials: ", len(study.trials))
        
        if interrupted: 
            exit(0)
        
        trial = study.best_trial
        best_trial_name = f"{self.config['realm_ai']['behavior_name']}_{trial.number}"
        print(f"\nBest trial: {best_trial_name}")

        if os.path.isdir('best_trial'):
            shutil.rmtree('./best_trial')
        os.mkdir('best_trial')
        shutil.copyfile(f"{best_trial_name}.yml", f"./best_trial/{best_trial_name}.yml")
        print(f'\nSaved {best_trial_name} to "best_trial" folder')    
        return best_trial_name