# REALM_AI RL-Subsystem

## Installation guide
In a terminal, enter the following commands:
1. `git clone https://github.com/realm-ai-project/RL-Subsystem.git`
2. `cd RL-Subsystem`
3. `pip install -e .`

## Quickstart guide 
1. Build an executable of the game.
2. In a terminal, do 
```
realm-tune --behavior-name <behavior_name> --env-path <path_to_environment>
```
where 
- `<behavior_name>` can be found under the agent's "Behavior Parameters" component in the Unity's inspector
- `<path_to_environment>` represents the path to the game executable 


## Notes
1. Does not support multiplayer environments (i.e., environments with >1 behaviour(s))
2. Does not support in-editor training, must be trained on a build
3. Resuming feature. If training is paused in the middle of hyperparameter tuning, the currently running trial will be discarded upon resuming. However, if training is paused in the middle of the full run, upon resuming it will automatically continue running the full run!
4. Error testing of specifying hyperparameters in yaml file is left to `mlagents` python package, which does a good job in checking if specified hyperparameters are valid. An important note is that it is not required to specify all hyperparameters, we can just have hyperparameters that we want to perform tuning over - others will automatically be defaulted. The only mandatory field is the trainer type (e.g., `ppo` or `sac`. Note: `realm-tune` is algorithm-agnostic).
5. The resuming feature works with or without wandb. If using wandb, we might see multiple runs with the same name.

## Known bugs
1. When continuing full run and using wandb, there will be multiple tensorboard files (due to resuming a run). As a result, when reporting the final result of the run to wandb, it is possible that the incorrect tensorboard file is read from.
