# load config file from toml file

import tomli
from collections import defaultdict
from moe import logger

from moe.model import ModelArgs

class Config:
    """
    A helper class to manage the configuration.
    Semantics:
    - Default config is loaded from a toml file. If no toml file is provided,
    then the default config is loaded from argparse defaults.
    - if toml file has missing keys, they are filled with argparse defaults.
    - if additional explicit cmd args are provided in addition to the toml
    file, they will override the toml config and the argparse defaults

    precedence order: cmdline > toml > argparse default

    Arg parsing semantics:

    Each argument starts with <prefix>_ which is the section name in the toml file
    followed by name of the option in the toml file. For ex,
    model.name translates to:
        [model]
        name
    in the toml file
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Load the configuration from the toml file.
        """
        config = {}
        try:
            with open(self.config_path, "rb") as f:
                for k, v in tomli.load(f).items():
                    config[k] = v;
        except (FileNotFoundError, tomli.TOMLDecodeError) as e:
            logger.error(f"Config file {self.config_path} not found, using default config")
            raise e


        for k, v in config.items():
            class_type = type(k.title(), (), v)
            setattr(self, k, class_type())
            print(f"Setting {k} to {v}")

        
        self._validate_config()
        return config

    def to_dict(self) -> dict:
        return self.config

    def _validate_config(self) -> None:
        assert self.model 
        self.model = ModelArgs(self.model)
        


if __name__ == "__main__":
    config = Config("./moe/config/tiny.toml")
    print(config)
