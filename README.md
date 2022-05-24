# REALM_AI RL-Subsystem

**Full documentation can be found [here](https://realm-ai-project.github.io/documentation/rl_subsystem/description/)**

## Installation guide
In a terminal, enter the following commands:
1. `git clone https://github.com/realm-ai-project/RL-Subsystem.git`
2. `cd RL-Subsystem`
3. `pip install -e .`

## Minimal quickstart guide 
1. Build an executable of the game.
2. In a terminal, do 
```
realm-tune --env-path <path_to_environment>
```
where 
- `<path_to_environment>` represents the path to the game executable 


## Notes
1. Does not support multiplayer environments (i.e., environments with >1 behaviour(s))
2. Does not support in-editor training, must be trained on a build
3. Resuming feature. If training is paused in the middle of hyperparameter tuning, the currently running trial will be discarded upon resuming. However, if training is paused in the middle of the full run, upon resuming it will automatically continue running the full run!
4. Error testing of specifying hyperparameters in yaml file is left to `mlagents` python package, which does a good job in checking if specified hyperparameters are valid. An important note is that it is not required to specify all hyperparameters, we can just have hyperparameters that we want to perform tuning over - others will automatically be defaulted. The only mandatory field is the trainer type (e.g., `ppo` or `sac`. Note: `realm-tune` is algorithm-agnostic).
5. The resuming feature works with or without wandb. If using wandb, we might see multiple runs with the same name.
6. Tested on wandb offline. When using wandb offline, after running to completion, `cd` into root folder of run (where wandb folder lies), and do `wandb sync --sync-all`. If there are any errors, delete those runs and retry the wandb sync command.
