import argparse
from logging import warning
import yaml
from copy import deepcopy
import os
import subprocess 
import time
import statistics
from enum import Enum, auto
import pickle
import shutil 

import requests
from tensorboard import program
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
import wandb

def parse_arguments():
    parser = argparse.ArgumentParser(description='Realm_AI hyperparameter optimization tool')
    parser.add_argument('--config_path', default='realm_tune/bayes.yaml')
    args = parser.parse_args()
    return args

class HpTuningType(Enum):
    CATEGORICAL = auto()
    UNIFORM_INT = auto()
    UNIFORM_FLOAT = auto()
    LOG_UNIFORM_FLOAT = auto()
    LOG_UNIFORM_INT = auto()

class OptunaHyperparamTuner:
    def __init__(self, config_file_path):
        self.load_config(config_file_path)
        self.find_hyperparameters_to_tune()
        self.tensorboard_url = None
    
    def load_config(self, path: str):
        def assert_config_structure():
            assert('realm_ai' in config and 'mlagents' in config)
            
            self.algo = config['realm_ai'].get('algorithm', 'bayes')
            assert self.algo in ['bayes', 'grid', 'random']
            if self.algo=='bayes':
                config['realm_ai']['warmup_trials'] = config['realm_ai'].get('warmup_trials', 5)
            
            if 'checkpoint_settings' not in config['mlagents']:
                config['mlagents']['checkpoint_settings'] = {}
            
            config['realm_ai']['eval_window_size'] = config['realm_ai'].get('eval_window_size', 1)
            
            assert isinstance(config['realm_ai']['eval_window_size'], int)

            self.use_wandb = 'wandb' in config['realm_ai'] and 'project' in config['realm_ai']['wandb'] and 'entity' in config['realm_ai']['wandb']
            if 'wandb' in config['realm_ai'] and 'offline' in config['realm_ai']['wandb'] and config['realm_ai']['wandb']['offline']:
                os.environ["WANDB_MODE"]="offline"
                raise NotImplementedError("Offline mode not supported for now!")

        try:
            with open(path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise Exception(f'Could not load configuration from {path}.')
        
        assert_config_structure()
        self.config = config

    def __call__(self, trial: optuna.trial.Trial) -> float:
        '''
        Optuna's objective function
        '''
        print(f'Running trial {trial.number}')

        run_id = f"{self.config['realm_ai']['behavior_name']}_{trial.number}"
        if self.use_wandb:
            wandb_run = wandb.init(entity=self.config['realm_ai']['wandb']['entity'], group=self.config['realm_ai']['folder_name'], project=self.config['realm_ai']['wandb']['project'], reinit=True)
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

        if self.tensorboard_url is None:
            self.__launch_tensorboard()

        score = self.__evaluate(run_id)
        print(f'Score for trial {trial.number}: {score}')

        if self.use_wandb:
            wandb.summary["Average Reward"] = score
            wandb.Api(overrides={'entity':self.config['realm_ai']['wandb']['entity']}).sync_tensorboard(root_dir=f"./results/{run_id}/{self.config['realm_ai']['behavior_name']}", run_id=wandb_run.id, project=self.config['realm_ai']['wandb']['project'])
            wandb_run.finish()

        return score

    def find_hyperparameters_to_tune(self):
        '''
        Find all hyperparameters to perform hyperparameter tuning on, 
        and store them in self.hyperparams_path and self.hyperparameters_to_tune
        '''
        hyperparameters = self.config['mlagents']['default_settings']
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
                    assert self.algo!='grid', f'Continuous hyperparameters "{k}" not allowed in grid search!'
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

    def __create_config_file(self, run_id, config):
        '''
        Create a config file for the given configuration
        '''
        config['checkpoint_settings']['run_id'] = run_id
        with open(f'{run_id}.yml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False) 

    def __launch_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", "results"])
        self.tensorboard_url = tb.launch()

    def __evaluate(self, run_id) -> int: 
        tb_query_url = f'{self.tensorboard_url}data/plugin/scalars/scalars'
        r = requests.get(tb_query_url, params={"run":f"{run_id}/{self.config['realm_ai']['behavior_name']}", "tag":"Environment/Cumulative Reward"})
        if r.status_code != requests.codes.ok:
            print(f"Error querying tensorboard on port 6006 for run_id:{run_id}")
            r.raise_for_status()
        response = r.json()
        _,_,cumulative_reward = zip(*response)
        return statistics.mean(cumulative_reward[-self.config['realm_ai']['eval_window_size']:])

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
        
        if self.use_wandb: # Delete existing runs under the same folder (group) on wandb if it already exists
            path = f"{self.config['realm_ai']['wandb']['entity']}/{self.config['realm_ai']['wandb']['project']}"
            runs_to_delete = wandb.Api().runs(path=path, filters={"group":self.folder}, per_page=1000)
            
            for run in runs_to_delete:
                run.delete()


        if os.path.isdir(self.folder):
            raise FileExistsError(f'"{self.folder}/" already exists, and no checkpoint exists in that folder. Please rename folder_name in config file, or delete the folder!')
        else:
            print('Creating folder named', self.folder)
            os.mkdir(self.folder)

        os.chdir(self.folder)

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
        # df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        if self.config['realm_ai']['total_trials'] >= len(new_study.trials):
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

def configure_for_full_run(config, best_trial_name):
    path = f"./best_trial/{best_trial_name}.yml"
    try:
        with open(path) as f:
            hyperparam = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise Exception(f'Could not load configuration from {path}.')
    # TODO: error checking for the following lines
    hyperparam['default_settings']['max_steps'] = config['max_steps']
    hyperparam['checkpoint_settings']['run_id'] = config['run_id']
    hyperparam['checkpoint_settings']['resume'] = True
    with open("./best_trial/full_run_config.yml", 'w') as f:
        yaml.dump(hyperparam, f, default_flow_style=False) 
    if os.path.isdir(f"./results/{config['run_id']}"):
        raise FileExistsError(f"Results for full run (./results/{config['run_id']}) already exist, program exiting...")
    shutil.copytree(f"./results/{best_trial_name}", f"./results/{config['run_id']}")

def main():
    args = parse_arguments()
    alg = OptunaHyperparamTuner(args.config_path)
    best_trial_name = alg.run()
    config = alg.config
    if 'full_run_after_tuning' in config['realm_ai']:
        configure_for_full_run(config['realm_ai']['full_run_after_tuning'], best_trial_name)

if __name__ == "__main__":
    main()

    
