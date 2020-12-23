from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass, field
import logging
import glob
import os
import hashlib
import json

import pandas as pd
import numpy as np
import yaml

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class ExperimentConfig(object):
    """
    Configuration container for a CDD experiment
    """

    experiment_id: str                 = field(repr=True, hash=True)
    # directory of columnar files
    data_path: str                     = field(repr=True, hash=None)
    output_dir: str                    = field(repr=True, hash=None)
    # num. stratified folds
    k: int                             = field(repr=True, hash=None, default=10)

    # config file
    config_file: Optional[str]         = field(repr=True, hash=False, default=None)
    # random seed
    seed: int                          = field(repr=True, default=42, hash=None)

    @staticmethod
    def from_file(config_file: str) -> "ExperimentConfig":
        """
        Loads a conf instance from a YAML file
        """
        cfp = os.path.abspath(config_file)
        with open(cfp) as f:
            return ExperimentConfig.from_str(config=f.read(), config_file_path=cfp)
  
    @staticmethod
    def from_str(config: str, config_file_path: Optional[str] = None) -> "ExperimentConfig":
        """
        Loads a conf instance from the contents (str) of a YAML file
        """
        params = yaml.load(config, Loader=yaml.FullLoader)
        #params.pop('preprocessing', None)
        # generate hash
        hash_obj = hashlib.sha256(json.dumps(params, sort_keys=True).encode())
        experiment_id = hash_obj.hexdigest()
        params["experiment_id"] = str(experiment_id)
        # add config file loc (if present)
        params["config_file"] = config_file_path
        return ExperimentConfig(**params)

    @property
    def short_sha(self) -> str:
      return self.experiment_id[:10]

