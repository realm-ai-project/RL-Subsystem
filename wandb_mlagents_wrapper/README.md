# WandB MLAgents Wrapper

## Usage
To use WandB with ML-Agents, simply define an extra field in the config file that is otherwise completely compatible with ML-Agents Python package.
To run, enter `wandb-mlagents-learn` in the terminal. This wrapper accepts the same cli arguments as `mlagents-learn`. This program can be used as a standalone passthrough for `mlagents-learn`.

## Supported arguments
```
wandb:
    project: <default: realm_ai>
    entity: <default: None>
    offline: <default: False>
    group: <default: None>
    jobtype: <default: None> # "job_type" is accepted too! 
```

## Note
- Resuming a run does not work (yet)