import yaml

class RealmTuneBaseConfig:
    def __init__(self, 
        behavior_name, 
        algorithm='bayes', 
        total_trials=5, 
        warmup_trials=3, 
        eval_window_size=1, 
        data_path=None,
        wandb={}
    ) -> None:
        self.behavior_name = behavior_name
        self.algorithm = algorithm
        self.total_trials = total_trials
        self.warmup_trials = warmup_trials
        self.eval_window_size = eval_window_size
        self.data_path = data_path
        self.wandb = wandb

    @staticmethod
    def from_dict(config_dict):
        return RealmTuneBaseConfig(**config_dict)


class MLAgentsBaseConfig:
    
    @staticmethod
    def from_dict(config_dict):
        return RealmTuneBaseConfig(**config_dict)

class RealmTuneConfig:
    @staticmethod
    def from_yaml_file(path_to_yaml_file): 
        with open(path_to_yaml_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return RealmTuneBaseConfig.from_dict(config)