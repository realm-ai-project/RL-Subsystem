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
from realm_tune.hyperparam_tuner import OptunaHyperparamTuner
from realm_tune.settings import RealmTuneConfig, WandBSettings
from realm_tune.cli_config import parser

class Runner:
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

        study = pickle.load( open( "optuna_study.pkl", "rb" ) )
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
            warning(f'{len(new_study.trials)} completed trials already found in folder "{os.getcwd()}". Exiting...')
            exit(0)
        print(f'Resuming from {len(new_study.trials)} completed trials')
        return new_study

    def _get_sampler(self)-> optuna.samplers.BaseSampler:
        if self.options.realm_ai.algorithm == 'bayes':
            return TPESampler(n_startup_trials=self.options.realm_ai.warmup_trials)
        elif self.options.realm_ai.algorithm == 'random':
            return RandomSampler()
        elif self.options.realm_ai.algorithm == 'grid':
            search_space = {i:v for i, _, v in self.options.mlagents.hyperparameters_to_tune}
            return GridSampler(search_space)

    def run(self):
        
        os.chdir(self.options.realm_ai.output_path)

        print('Results will be stored in', os.getcwd())

        if os.path.isfile("./optuna_study.pkl"):
            study = self._restore_from_checkpoint()
        else:
            study = self._run_from_scratch()
        
        hyperparam_tuner = OptunaHyperparamTuner(self.options)
        interrupted = False
        try:
            study.optimize(hyperparam_tuner, n_trials=self.options.realm_ai.total_trials-len(study.trials))
        except KeyboardInterrupt:
            interrupted = True

        pickle.dump(study, open( "optuna_study.pkl", "wb" ) )
        print('Saved study as optuna_study.pkl')

        print("Number of finished trials: ", len(study.trials))
        
        if interrupted: 
            exit(0)
        
        trial = study.best_trial
        best_trial_name = f"{self.options.realm_ai.behavior_name}_{trial.number}"
        print(f"\nBest trial: {best_trial_name}")

        if os.path.isdir('best_trial'):
            shutil.rmtree('./best_trial')
        os.mkdir('best_trial')
        shutil.copyfile(f"{best_trial_name}.yml", f"./best_trial/{best_trial_name}.yml")
        print(f'\nSaved {best_trial_name} to "best_trial" folder')    
        return best_trial_name

def main(args=None):
    runner = Runner.from_argparse(parser.parse_args(args))
    runner.run()

if __name__ == "__main__":
    main(["--config-path","realm_tune/bayes.yaml",])
