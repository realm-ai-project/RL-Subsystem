# WandB MLAgents Wrapper

## Usage
To use WandB with ML-Agents, simply define an extra field in the config file that is otherwise completely compatible with ML-Agents Python package.

## Supported arguments
```
wandb:
    project: <default: realm_ai>
    entity: <default: None>
    offline: <default: False>
    group: <default: None>
    job_type: <default: None>
```

## Note
- Resuming a run does not work (yet)