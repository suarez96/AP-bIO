from train_utils import get_rolling_windows
import os
from train_utils import evaluate, train, export, make_train_test_sets, build_model_name, predict
from scipy.signal import detrend
from Data import MarshData
import Transforms
import os
import numpy as np

import yaml

def load_yaml(path):

    with open(path) as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    
    return parsed_yaml

def run(
    cmd_args
):
    """
    Main app function. Runs data loading, training and evaluation
    Parameters:
    cmd_args (Argparse arguments): set of arguments that are given by the command line
    interface when starting the "main" script. Refers to things that are unlikely to change
    between training runs on the same computer, ie. the path to the yaml params and the path
    to the training data. Training hyperparameters are set in the yaml parameter file.
    """

    # separate function?
    yaml_args = load_yaml(cmd_args.yaml_path)
    transform_hparams = parse_hparams_from_yaml(pipeline)
    train_X_transforms = Transforms.build_pipeline(yaml_args['ecg_pipeline'], pipeline_args=transform_hparams)
    train_y_transforms = Transforms.build_pipeline(yaml_args['ip_pipeline'], pipeline_args=transform_hparams)
    
    # X_train, y_train
    X_ecg_rolling_train_stack_np, y_ip_train_stack_np = None, None
    
    print(vars(cmd_args))