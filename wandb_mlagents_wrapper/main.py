import sys
import re
from copy import deepcopy
import os

from mlagents.trainers.learn import run_cli, parse_command_line
import wandb
import yaml


class WandBMLAgentsWrapper:
    def __init__(self, arguments) -> None:
        self.arguments = arguments
        self.use_wandb = False
        self.__parse_arguments()

    def __parse_arguments(self):
        # Check if config file is passed in as an argument
        regex = re.compile('(.)+(\.yaml|\.yml)')
        config_file = list(filter(regex.match, self.arguments))
        assert len(config_file)<=1, 'Must not pass in more than one config file in the cli arguments!'
        if config_file:
            self.__parse_wandb_config(config_file[0])

    def __parse_wandb_config(self, config_file):
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if 'wandb' in config:
            self.use_wandb = True
            wandb_config = config['wandb']
            self.project =  wandb_config.get('project', 'realm_ai')
            self.entity = wandb_config.get('entity', None)
            self.offline = wandb_config.get('offline', False)
            self.group = wandb_config.get('group', None)
            self.job_type = wandb_config.get('job_type', None)
            self.__extract_wandb_from_config(config, config_file)
    
    def __extract_wandb_from_config(self, config, config_file_name):
        '''
        Make config file compatible with ML-Agents by extracting out wandb configurations
        '''
        if 'wandb' in config:
            config = deepcopy(config)
            del config['wandb']
            with open(config_file_name, 'w') as f:
                yaml.dump(config, f, default_flow_style=False) 
    
    def run_training(self):
        mlagents_config = parse_command_line(argv=self.arguments[1:])

        if self.use_wandb:
            if self.offline:
                os.environ["WANDB_MODE"]="offline"
            wandb_run = wandb.init(entity=self.entity, group=self.group, project=self.project, sync_tensorboard=True, job_type=self.job_type, )
            # resume=True if mlagents_config.checkpoint_settings.resume else "allow")
            wandb_run.name = mlagents_config.checkpoint_settings.run_id
        
        run_cli(mlagents_config)

def main():
    mlagents = WandBMLAgentsWrapper(sys.argv)    
    mlagents.run_training()

if __name__=='__main__':
    main()