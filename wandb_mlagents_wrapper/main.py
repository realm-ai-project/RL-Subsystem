import sys
import re
from copy import deepcopy
import os
import tempfile
import warnings

from mlagents.trainers.learn import run_cli, parse_command_line
import wandb
import yaml


class WandBMLAgentsWrapper:
    def __init__(self, arguments) -> None:
        self.arguments = arguments
        # Set default values
        self.use_wandb = False
        self.project =  'realm_ai'
        self.entity = None
        self.offline = False
        self.group = None
        self.job_type = None
        self.__parse_arguments()

    def __parse_arguments(self):
        # Check if config file is passed in as an argument
        regex = re.compile('(.)+(\.yaml|\.yml)')
        config_file = list(filter(regex.match, self.arguments))
        assert len(config_file)<=1, 'Must not pass in more than one config file in the cli arguments!'
        if config_file:
            self.__parse_wandb_config(config_file[0])
        
        # Check if resume flag is passed in -- we don't support resume yet
        if "--resume" in self.arguments:
            warnings.warn("Wrapper does not support resume yet, it will be treated as a new run on WandB!")

    def __parse_wandb_config(self, config_file):
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        if 'wandb' in config:
            self.use_wandb = True
            wandb_config = config['wandb']
            if wandb_config is not None:
                self.project =  wandb_config.get('project', self.project)
                self.entity = wandb_config.get('entity', self.entity)
                self.offline = wandb_config.get('offline', self.offline)
                self.group = wandb_config.get('group', self.group)
                self.job_type = wandb_config['jobtype'] if 'jobtype' in wandb_config else wandb_config.get('job_type', self.job_type)
            self.__create_temp_file(config, config_file)
    
    def __create_temp_file(self, config, config_file_name):
        '''
        Create temporary config file compatible with ML-Agents by extracting out wandb configurations
        '''
        if self.use_wandb:
            config = deepcopy(config)
            del config['wandb']
            fd, self.temp_filename = tempfile.mkstemp(suffix='.yml')
            with open(fd, 'w') as f:
                yaml.dump(config, f, default_flow_style=False) 
            # Override argument with temp file
            self.arguments[self.arguments.index(config_file_name)] = self.temp_filename
    
    def run_training(self):
        mlagents_config = parse_command_line(argv=self.arguments[1:])

        if self.use_wandb:
            if self.offline:
                os.environ["WANDB_MODE"]="offline"
            wandb_run = wandb.init(entity=self.entity, group=self.group, project=self.project, sync_tensorboard=True, job_type=self.job_type,)
            wandb_run.config.update(mlagents_config.as_dict())
            # resume=True if mlagents_config.checkpoint_settings.resume else "allow")
            wandb_run.name = mlagents_config.checkpoint_settings.run_id
        
        run_cli(mlagents_config)

        if self.use_wandb:
            os.remove(self.temp_filename)

def main():
    mlagents = WandBMLAgentsWrapper(sys.argv)    
    mlagents.run_training()

if __name__=='__main__':
    main()