import argparse
import yaml
from copy import deepcopy
import os
import subprocess 
import uuid
import statistics

import requests
from tensorboard import program

def parse_arguments():
    parser = argparse.ArgumentParser(description='Realm_AI hyperparameter optimization tool')
    parser.add_argument('--config_path', default='RLSubsystem/GridSearch/test_config.yaml')
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

class GridSearchAlgorithm:
    def __init__(self, config):
        self.config = config
        self.find_hyperparameters_to_tune()
        self.generate_all_combinations()
        num_combinations = len(self.configs)
        self.run_ids = [f"{config['realm_ai']['behavior_name']}_{i}" for i in range(num_combinations)]

    def find_hyperparameters_to_tune(self):
        '''
        Find all hyperparameters to perform grid search on, 
        and store them in self.hyperparams_path and self.hyperparameters_to_tune
        '''
        hyperparameters = self.config['mlagents']['default_settings']
        # tells us path to hyperparameter in config file given hyperparameter name
        self.hyperparams_path = {}
        # a tuple of two values (hyperparameter, list of values)
        self.hyperparameters_to_tune = []

        def parse_recursively(dict_, path: list):
            if not dict_:
                return 
            for k, v in dict_.items():
                # if its a list, its a hyperparameter to tune
                if isinstance(v, list):
                    self.hyperparameters_to_tune.append((k, v))
                    if k in self.hyperparams_path:
                        raise Exception(f'Duplicated hyperparameters found!, hyperparameter: {k}')
                    self.hyperparams_path[k] = path+[k]
                # recursively parse nested dictionary
                elif isinstance(v, dict):
                    parse_recursively(dict_[k], path+[k])
        parse_recursively(hyperparameters, list())
        
    
    def generate_all_combinations(self):
        '''
        Generate all possible permutation of configurations 
        and store them in self.configs
        '''
        hyperparameters = self.config['mlagents']
        self.configs = []
        
        def generate_recursively(curr_config, hyperparams_to_tune):
            if not hyperparams_to_tune:
                self.configs.append(curr_config)
                return
            hyperparam, values = hyperparams_to_tune[0]
            for val in values:
                config = deepcopy(curr_config)
                temp = config['default_settings']
                path = self.hyperparams_path[hyperparam]
                for p in path[:-1]:
                    temp = temp[p]
                temp[path[-1]] = val
                generate_recursively(config, hyperparams_to_tune[1:])
        generate_recursively(hyperparameters, self.hyperparameters_to_tune)


    def __create_config_files(self):
        '''
        Create a config file for every configuration
        '''
        self.folder = f'runs-{uuid.uuid4().hex[:6]}'
        while os.path.isdir(self.folder):
            self.folder = f'runs-{uuid.uuid4().hex[:6]}'
        
        print('Creating folder named', self.folder)
        
        os.mkdir(self.folder)

        os.chdir(self.folder)

        for run_id, config in zip(self.run_ids, self.configs):
            config['checkpoint_settings']['run_id'] = run_id
            with open(f'{run_id}.yml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False) 

    def __launch_tensorboard(self):
        # self.tensorboard = subprocess.Popen(["tensorboard", "--logdir=results", "--port=6006"])    
        tb = program.TensorBoard()
        tb.configure(argv=[None, "--logdir", "results"])
        self.tensorboard_url = tb.launch()

    def __evaluate(self, run_id) -> int: 
        tb_query_url = f'{self.tensorboard_url}data/plugin/scalars/scalars'
        r = requests.get(tb_query_url, params={"run":f"{run_id}/{self.config['realm_ai']['behavior_name']}", "tag":"Environment/Cumulative Reward"})
        if r.status_code != requests.codes.ok:
            raise Exception(f"Error querying tensorboard on port 6006 for run_id:{run_id}")
        response = r.json()
        _,_,cumulative_reward = zip(*response)
        return statistics.mean(cumulative_reward[-5:])


    def run(self):
        '''
        1) Create config files
        2) Run mlagents sequentially for every configuration
        3) Evaluate all results, and return best empirical configuration
        '''
        self.__create_config_files()

        for run_id in self.run_ids:
            subprocess.run(["mlagents-learn", f"{run_id}.yml"])

        self.__launch_tensorboard()

        self.scores = []
        for run_id in self.run_ids:
            self.scores.append((run_id, self.__evaluate(run_id)))
        self.scores.sort(key=lambda x: x[1], reverse=True)

        print(f"Run with highest score: {self.scores[0][0]}")



if __name__=='__main__':
    args = parse_arguments()
    config = load_config(args.config_path)
    alg = GridSearchAlgorithm(config)
    alg.run()
