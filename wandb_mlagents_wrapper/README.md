# WandB MLAgents Wrapper

## Usage
To use WandB with ML-Agents, simply add an extra "wandb" field in a config file that is otherwise completely compatible with ML-Agents Python package.
To run, enter `wandb-mlagents-learn` in the terminal. This wrapper accepts the same cli arguments as `mlagents-learn`. This program can be used as a standalone passthrough for `mlagents-learn`.

## Supported arguments
At minimum, here is what the `.yaml` config file should contain:
```
wandb:
    project: <default: realm_ai>
    entity: <default: None>
    offline: <default: False>
    group: <default: None>
    jobtype: <default: None> # "job_type" is accepted too! 
```

## Note
- Resuming a run on wandb does not work (yet)
- If we intend to use mlagents default parameters, it is okay to pass in a config file that solely contains wandb config (as shown above)
- If "wandb" is not defined in the config file, wandb will not be used, and functionality will then be identical to mlagents!
- When resuming runs, multiple tensorboard files may be present. Currently, this wrapper arbitrarily uses one of them to report the final reward to WandB