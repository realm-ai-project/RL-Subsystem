from setuptools import setup, find_packages

setup(
    name='realm_ai',
    version='0.0.1',
    description="REALM_AI's RL Subsystem",
    install_requires=[
        "requests",  
        "mlagents",
        "optuna",
    ],
    python_requires=">=3.6.1",
    packages=find_packages(include=['realm_train', 'realm_tune']),
    entry_points={
        "console_scripts": [
            "realm-tune=realm_tune.main:main",
            "realm-train=realm_train.main:main",
        ]
    },
)
