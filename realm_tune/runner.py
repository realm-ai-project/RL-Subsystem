import logging
import os
import time
import pickle
from logging import warning
import shutil
import cattr
import argparse
import warnings

import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
import yaml

from realm_tune.hyperparam_tuner import OptunaHyperparamTuner
from realm_tune.settings import FullRunConfig, RealmTuneConfig, WandBSettings, load_yaml_file
from realm_tune.cli_config import parser
from realm_tune.utils import add_wandb_config
from wandb_mlagents_wrapper.main import WandBMLAgentsWrapper

class Runner:
    NAME_OF_FULL_RUN = "best_run"
    CONFIG_SAVE_DIR = "./trial_config"
    BEST_TRIAL_FOLDER_NAME = "best_trial"
    BEST_TRIAL_DIR = os.path.join(CONFIG_SAVE_DIR, BEST_TRIAL_FOLDER_NAME)
    OPTUNA_STUDY_CKPT_NAME = "tuning_checkpoint.pkl"

    def __init__(self, options: RealmTuneConfig):
        self.options: RealmTuneConfig = options

    @staticmethod
    def from_argparse(args: argparse.Namespace):
        options = RealmTuneConfig.from_argparse(args)
        return Runner(options)

    def _run_from_scratch(self) -> optuna.Study:
        assert len(os.listdir('.'))==0, f'{self.options.realm_ai.output_path} is not empty but checkpoint file not found. Please delete folder and try again'
        
        if self.options.realm_ai.wandb.use_wandb:
            wandb_metadata = self.options.realm_ai.wandb.to_dict()
            pickle.dump(wandb_metadata, open( "wandb_metadata.pkl", "wb" ) )

        sampler = self._get_sampler()
        study = optuna.create_study(study_name=self.options.realm_ai.output_path, sampler=sampler, direction="maximize")
        return study

    def _restore_from_checkpoint(self) -> optuna.Study:
        sampler = self._get_sampler()
        new_study = optuna.create_study(sampler=sampler, direction="maximize")

        study = pickle.load( open( self.OPTUNA_STUDY_CKPT_NAME, "rb" ) )
        for trial in study.trials:
            if trial.state.is_finished():
                new_study.add_trial(trial)
        
        if self.options.realm_ai.wandb.use_wandb:
            if not os.path.isfile("wandb_metadata.pkl"):
                warning("Restoring from checkpoint but wandb metadata not found!")
            else:
                metadata = pickle.load(open("wandb_metadata.pkl", "rb"))
                self.options.realm_ai.wandb = WandBSettings.structure(metadata, None)

        # df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        if self.options.realm_ai.total_trials <= len(new_study.trials):
            warning(f'{len(new_study.trials)} completed trials already found in folder "{os.getcwd()}"')
        else: print(f'Resuming from {len(new_study.trials)} completed trials')
        return new_study

    def _get_sampler(self)-> optuna.samplers.BaseSampler:
        if self.options.realm_ai.algorithm == 'bayes':
            return TPESampler(n_startup_trials=self.options.realm_ai.warmup_trials)
        elif self.options.realm_ai.algorithm == 'random':
            return RandomSampler()
        elif self.options.realm_ai.algorithm == 'grid':
            search_space = {i:v for i, _, v in self.options.mlagents.hyperparameters_to_tune}
            return GridSampler(search_space)

    def _save_best_trial(self, study):
        trial = study.best_trial
        best_trial_name = f"{self.options.realm_ai.behavior_name}_{trial.number}"
        print(f"\nBest trial: {best_trial_name}")

        os.makedirs(self.BEST_TRIAL_DIR, exist_ok=True)
        shutil.copyfile(os.path.join(self.CONFIG_SAVE_DIR, f"{best_trial_name}.yml"), os.path.join(self.BEST_TRIAL_DIR, f"{best_trial_name}.yml"))
        print(f'\nSaved {best_trial_name} to "best_trial" folder') 
        return best_trial_name

    def run_hyperparameter_tuning(self):

        if os.path.isfile(self.OPTUNA_STUDY_CKPT_NAME):
            study = self._restore_from_checkpoint()
        else:
            study = self._run_from_scratch()
        
        hyperparam_tuner = OptunaHyperparamTuner(self.options, config_save_path=self.CONFIG_SAVE_DIR)
        interrupted = False
        try:
            study.optimize(hyperparam_tuner, n_trials=self.options.realm_ai.total_trials-len(study.trials))
        except KeyboardInterrupt:
            interrupted = True

        pickle.dump(study, open( self.OPTUNA_STUDY_CKPT_NAME, "wb" ) )
        print(f'Saved study as {self.OPTUNA_STUDY_CKPT_NAME}')

        print("Number of finished trials: ", len(study.trials))
        
        if interrupted: 
            exit(0)
           
        best_trial_name = self._save_best_trial(study)
        return best_trial_name

    def _create_full_run_config(self, best_trial_name:str, config:FullRunConfig):        
        path = os.path.join(self.BEST_TRIAL_DIR, f"{best_trial_name}.yml")
        try:
            hyperparam = load_yaml_file(path)
        except FileNotFoundError:
            raise Exception(f'Could not load configuration from {path}.')

        hyperparam['default_settings']['max_steps'] = config.full_run_max_steps
        hyperparam['checkpoint_settings']['run_id'] = self.NAME_OF_FULL_RUN
        hyperparam['checkpoint_settings']['resume'] = True
        
        if self.options.realm_ai.wandb.use_wandb:
            add_wandb_config(hyperparam, self.options.realm_ai.wandb)
        
        with open(os.path.join(self.BEST_TRIAL_DIR, f"{self.NAME_OF_FULL_RUN}_config.yml"), 'w') as f:
            yaml.dump(hyperparam, f, default_flow_style=False) 
        
        if not os.path.isdir(f"./results/{self.NAME_OF_FULL_RUN}"):
            shutil.copytree(f"./results/{best_trial_name}", f"./results/{self.NAME_OF_FULL_RUN}")
        else:
            warning(f'Results for full run (./results/{self.NAME_OF_FULL_RUN}) already exist, resuming full run...')

    def _run_full_run_after_tuning(self, best_trial_name:str):
        self._create_full_run_config(best_trial_name, self.options.realm_ai.full_run_after_tuning)

        # Run it directly rather than in a subprocess so that interrupts are properly caught by mlagents-learn
        WandBMLAgentsWrapper(['wandb-mlagents-learn', os.path.join(self.BEST_TRIAL_DIR, f"{self.NAME_OF_FULL_RUN}_config.yml")]).run_training()

    def run(self):
        os.chdir(self.options.realm_ai.output_path)
        print('Results will be stored in', os.getcwd())

        best_trial_name = self.run_hyperparameter_tuning()

        if self.options.realm_ai.full_run_after_tuning.full_run:
            self._run_full_run_after_tuning(best_trial_name)    

def main(args=None):
    runner = Runner.from_argparse(parser.parse_args(args))
    runner.run()

if __name__ == "__main__":
    main(["--config-path","realm_tune/bayes.yaml",])
