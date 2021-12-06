import logging
import os
import time
import pickle
from logging import warning
import shutil

import optuna

from realm_tune.settings import RealmTuneConfig

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

def run_cli(options: RealmTuneConfig):
    if not os.path.isdir(options.realm_ai.output_path):
        os.makedirs(options.realm_ai.output_path)
    os.chdir(options.realm_ai.output_path)

    print('Results will be stored in', os.getcwd())

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

def main():
    args = parse_arguments()
    alg = OptunaHyperparamTuner(args)
    best_trial_name = alg.run()
    config = alg.config

if __name__ == "__main__":
    main()
