import argparse
import os
from typing import Dict, Optional, Set, Union


# From mlagents
class DetectDefault(argparse.Action):
    """
    Internal custom Action to help detect arguments that aren't default.
    """

    non_default_args: Set[str] = set()

    def __call__(self, arg_parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        DetectDefault.non_default_args.add(self.dest)

# From mlagents
class DetectDefaultStoreTrue(DetectDefault):
    """
    Internal class to help detect arguments that aren't default.
    Used for store_true arguments.
    """

    def __init__(self, nargs=0, **kwargs):
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, arg_parser, namespace, values, option_string=None):
        super().__call__(arg_parser, namespace, True, option_string)

def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Realm_AI hyperparameter optimization tool')
    parser.add_argument('--config-path', default=f'{os.path.dirname(os.path.realpath(__file__))}/default.yaml', action=DetectDefault)
    parser.add_argument('--output-path', type=str, default=None, help="Specify path where data is stored", action=DetectDefault)
    parser.add_argument('--behavior-name', type=str, default=None, help='Name of behaviour. This can be found under the agent\'s "Behavior Parameters" component in the inspector of Unity', action=DetectDefault)
    parser.add_argument('--algorithm', type=str, default='bayes', choices=['bayes', 'random', 'grid'], help="Algorithm for hyperparameter tuning", action=DetectDefault)
    parser.add_argument('--total-trials', type=int, default=10, help="Number of trials", action=DetectDefault)
    parser.add_argument('--warmup-trials', type=int, default=5, help="Number of warmup trials (only works for bayes algorithm)", action=DetectDefault)
    parser.add_argument('--eval-window-size', type=int, default=1, help="Training run is evaluated by taking the average eps rew of past x episodes", action=DetectDefault)
    parser.add_argument('--env-path', type=str, default=None, help="Path to environment. If specified, overrides env_path in the config file", action=DetectDefault)

    wandb_config = parser.add_argument_group(title="Weights and Biases Configuration")
    # Have an explicit field just so that if users want to keep everything default
    wandb_config.add_argument('--use-wandb', action=DetectDefaultStoreTrue)
    # If user uses any of the following wandb fields, it automatically infers that they intend to use wandb!
    wandb_config.add_argument('--wandb-project', type=str, action=DetectDefault, default='realm_tune')
    wandb_config.add_argument('--wandb-entity', type=str, default=None, action=DetectDefault)
    wandb_config.add_argument('--wandb-offline', action=DetectDefaultStoreTrue)
    wandb_config.add_argument('--wandb-group', type=str, default=None, action=DetectDefault)
    wandb_config.add_argument('--wandb-jobtype', type=str, default=None, action=DetectDefault)

    return parser

parser = _create_parser()

