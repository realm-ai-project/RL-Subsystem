from setuptools import setup, find_packages

setup(
    name='realm_ai',
    version='0.0.1',
    description="REALM_AI's RL Subsystem",
    install_requires=[
        "requests",  
        "mlagents",
        "optuna",
        "wandb",
        "tensorflow>=2.0.0" # Not tested
    ],
    python_requires=">=3.7", # tested with python 3.8
    packages=find_packages(include=['realm_train', 'realm_tune', 'wandb_mlagents_wrapper']),
    package_data={'realm_tune': ['default.yaml']},
    entry_points={
        "console_scripts": [
            "realm-tune=realm_tune.runner:main",
            # "realm-train=realm_train.main:main",
            "wandb-mlagents-learn=wandb_mlagents_wrapper.main:main",
        ]
    },
)
