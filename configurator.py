"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

from config import Config


def get_config(config_name: str) -> Config:
    match config_name:
        case "eval_gpt2_large":
            from configs.eval_gpt2_large import config
        case "eval_gpt2_medium":
            from configs.eval_gpt2_medium import config
        case "eval_gpt2_xl":
            from configs.eval_gpt2_xl import config
        case "eval_gpt2":
            from configs.eval_gpt2 import config
        case "finetune_shakespeare":
            from configs.finetune_shakespeare import config
        case "train_gpt2":
            from configs.train_gpt2 import config
        case "train_shakespeare_char":
            from configs.train_shakespeare_char import config
        case _:
            raise ValueError(f"Unknown config name: {config_name}")

    return config


def get_config_from_args(args: list[str] | None = None, config: Config | None = None) -> Config:
    """Process command line arguments to build a config.
    We first load the base config, then override it with CLI arguments.
    
    Args:
        args: List of command line arguments to process
        
    Returns:
        Config object with processed arguments
    """
    if args is None:
        args = sys.argv[1:]

    if config is None:
        config = Config()

    for arg in args:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            if arg.endswith('.py'):
                arg = arg[:-3]
            if arg.startswith('config/'):
                arg = arg[7:]
            loaded_config = get_config(arg)
            config.update(loaded_config)
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            key, val = arg.split('=')
            key = key[2:]
            if key in config:
                try:
                    # attempt to eval it it (e.g. if bool, number, or etc)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok
                assert type(attempt) == type(config[key])
                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                config[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")
                
    return config
