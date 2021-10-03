import argparse
import yaml
from copy import deepcopy
import os
import subprocess 
import uuid
import statistics
from enum import Enum, auto
import pickle

import requests
from tensorboard import program
import optuna
from optuna.samplers import TPESampler

def parse_arguments():
    parser = argparse.ArgumentParser(description='Realm_AI hyperparameter optimization tool')
    parser.add_argument('--config_path', default='BayesianOpt/test_config.yaml')
    args = parser.parse_args()
    return args

def load_config(path: str):
    def assert_config_structure():
        assert('realm_ai' in config and 'mlagents' in config)
    
    try:
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise Exception(f'Could not load configuration from {path}.')
    
    assert_config_structure()
    return config

class HpTuningType(Enum):
    CATEGORICAL = auto()
    UNIFORM_INT = auto()
    UNIFORM_FLOAT = auto()
    LOG_UNIFORM_FLOAT = auto()
    LOG_UNIFORM_INT = auto()

class BayesianOptimAlgorithm:
    def __init__(self, config):
        self.config = config
        self.find_hyperparameters_to_tune()
        self.tensorboard_url = None
    
    def __call__(self, trial: optuna.trial.Trial) -> float:
        '''
        Optuna's objective function
        '''
        print(f'Running trial {trial.number}')
        
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
        
        run_id = f"{self.config['realm_ai']['behavior_name']}_{trial.number}"

        self.__create_config_file(run_id, curr_config)

        subprocess.run(["mlagents-learn", f"{run_id}.yml", "--force"])

        if self.tensorboard_url is None:
            self.__launch_tensorboard()

        score = self.__evaluate(run_id)
        print(f'Score for trial {trial.number}: {score}')
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
        r = requests.get(tb_query_url, params={"run":f"{run_id}/{config['realm_ai']['behavior_name']}", "tag":"Environment/Cumulative Reward"})
        if r.status_code != requests.codes.ok:
            raise Exception(f"Error querying tensorboard on port 6006 for run_id:{run_id}")
        response = r.json()
        _,_,cumulative_reward = zip(*response)
        return statistics.mean(cumulative_reward[-int(self.config['realm_ai'].get('eval_window_size', 1)):])

    def run_from_scratch(self, run_id=None) -> optuna.Study:
        self.folder = run_id
        while self.folder is None or os.path.isdir(self.folder):
            self.folder = f'runs-{uuid.uuid4().hex[:6]}'
        
        print('Creating folder named', self.folder)
        
        os.mkdir(self.folder)

        os.chdir(self.folder)

        sampler = TPESampler(n_startup_trials=self.config['realm_ai']['warmup_trials'])
        study = optuna.create_study(study_name=self.folder, sampler=sampler, direction="maximize")
        return study

    def restore_from_checkpoint(self, run_id) -> optuna.Study:
        self.folder = run_id
        assert(os.path.isdir(self.folder))

        os.chdir(self.folder)

        sampler = TPESampler(n_startup_trials=self.config['realm_ai']['warmup_trials'])
        new_study = optuna.create_study(study_name=self.folder, sampler=sampler, direction="maximize")
        study = pickle.load( open( "optuna_study.pkl", "rb" ) )
        for trial in study.trials:
            if trial.state.is_finished():
                new_study.add_trial(trial)
        # df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        print(f'Resuming from {len(new_study.trials)} completed trials')
        return new_study

    def run(self):
        '''
        1) Create and cd into new/existing folder 
        2) Create a new Optuna study using the TPESampler, or restore previously saved study
        '''
        if 'run_id' in self.config['realm_ai'] and os.path.isdir(self.config['realm_ai']['run_id']) and os.path.isfile(f"{self.config['realm_ai']['run_id']}/optuna_study.pkl"):
            study = self.restore_from_checkpoint(self.config['realm_ai']['run_id'])
        else:
            study = self.run_from_scratch(run_id=self.config['realm_ai']['run_id'] if 'run_id' in self.config['realm_ai'] else None)
        
        try:
            study.optimize(self, n_trials=self.config['realm_ai']['num_trials'])
        except KeyboardInterrupt:
            pass

        pickle.dump(study, open( "optuna_study.pkl", "wb" ) )
        print('Saved study as optuna_study.pkl')

        print("Number of finished trials: ", len(study.trials))

        trial = study.best_trial
        print(f"Best trial: {self.config['realm_ai']['behavior_name']}_{trial.number}")

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print("    {}: {}".format(key, value))

    


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.config_path)
    alg = BayesianOptimAlgorithm(config)
    alg.run()

    
