from Data import MarshData, FantasiaData
from Transforms import WPC
import Transforms
from torch.utils.data import DataLoader, TensorDataset
import torch
from Signal import Signal
from train_utils import get_rolling_windows
from tsai.all import get_splits, TSRegression, TSStandardize, get_ts_dls
import os
import numpy as np
from tqdm import tqdm
from operator import itemgetter
import constants

import logging
logger = logging.getLogger(__name__)

def build_ECG_input_windows(
        args: dict,
        dataset
    ):
    """
    Build the rolling windows from the ECG data inside each of the subject
    signals.
    Dataset: a list of MarshData or FantasiaData objects to iterate through

    """

    # matrices to feed to the dataloader
    X_ecg_rolling_train_stack = []
    y_ip_train_stack = []
    num_windows_per_subject = []

    # yaml args define the params of each transform
    global_ecg_pipeline = Transforms.build_transforms(
        args['yaml_args']['global_ecg_pipeline']
    )
    global_ip_pipeline = Transforms.build_transforms(
        args['yaml_args']['global_ip_pipeline']
    )

    # take global ECG and break it into parts after applying transform
    for subject in tqdm(dataset, desc="building dataloader..."):
        
        subject_id = int(subject.ECG().filepath.split('/')[-2])

        input_ecg_raw = subject.ECG().transform(transforms=global_ecg_pipeline)
        input_ip_raw = subject.IP().transform(transforms=global_ip_pipeline)

        # extract transformed data
        input_ecg = input_ecg_raw.transformed_data
        input_ip = input_ip_raw.transformed_data

        # get rolling windows
        X_ecg_rolling = get_rolling_windows(
            input_ecg, 
            window_size=args['yaml_args']['model_params']['seq_len'], 
            jump=args['yaml_args']['learner_params']['jump_size']
        )
        y_ip = input_ip.data[args['yaml_args']['model_params']['seq_len']-1:][::args['yaml_args']['learner_params']['jump_size']]

        X_ecg_rolling_train_stack.append(X_ecg_rolling)
        y_ip_train_stack.append(y_ip)
        num_windows_per_subject.append(len(X_ecg_rolling))

    return np.vstack(X_ecg_rolling_train_stack), np.stack(y_ip_train_stack).flatten(), num_windows_per_subject


def loader_from_dataset(
    args,
    dataset,
    batch_tfms = TSStandardize(by_sample=True), # Standardize data TODO add more batch transforms
    valid_size = 0.2,
    shuffle=True,
    torch_loader=False
):

    """
    Construct the dataloader based on a train or test list of MarshData/FantasiaData objects.
    shuffle: shuffle data in the loader
    """
    X_ecg_rolling_stack_np, y_ip_stack_np, num_windows_per_subject = build_ECG_input_windows(args=args, dataset=dataset)
    X_array = X_ecg_rolling_stack_np.reshape(-1, 1, args['yaml_args']['model_params']['seq_len'])
    y_array = y_ip_stack_np.reshape(-1, 1)
    assert X_array.shape[0] == y_array.shape[0], "Inputs and Target shapes do not match!"
    
    # use pytorch dataloader
    # for evaluation. Avoids the strange index error 
    if torch_loader:
        X_tensor = torch.tensor(X_array).float()  # Convert to float32 tensor
        y_tensor = torch.tensor(y_array).float()  # Convert to float32 tensor
        dls = DataLoader(
            TensorDataset(X_tensor, y_tensor), 
            batch_size=args['yaml_args']['learner_params']['batch_size']
        )
    # use fastai type dataloader
    else:
        splits = get_splits(
            y_array, 
            valid_size=valid_size, 
            stratify=True, random_state=23, shuffle=shuffle, show_plot=False
        )
        # TODO investigate this step
        tfms  = [None, TSRegression()]
        dls = get_ts_dls(
            X_array, 
            y_array, 
            splits=None if not valid_size else splits, 
            tfms=tfms, 
            batch_tfms=batch_tfms, 
            bs=args['yaml_args']['learner_params']['batch_size']
        )
    return dls, num_windows_per_subject

def build_loaders(args, train: bool=True, test: bool=False, shuffle_test: bool=False, test_idxs: list=None):

    """
    test: whether to build a test dataloader or return None
    shuffle_test (bool): whether to shuffle the data in the test loader. usuallly false because we want to build the predictions
    in time order
    """

    dataset = {
        int(i): MarshData(os.path.join(args['marsh_path'], i), verbose=False) for i in tqdm(os.listdir(args['marsh_path']), desc='traversing MARSH data...') if len(i) == 4
    }
    assert train or test, "One of 'train' or 'test' must be set to true"

    if train:
        train_idxs = constants.argsort_subject_ids[:int(args['yaml_args']['data']['train_samples'])]
        
        # TODO add shuffle or tracking for subjects by ID
        train_dataset = itemgetter(*train_idxs)(dataset)
        if len(train_idxs) == 1:
            train_dataset = tuple([train_dataset])
        train_dataloader, n_win_sub_train = loader_from_dataset(args=args, dataset=train_dataset)
    
    else:
        train_dataloader, n_win_sub_train = None, None

    # load test indices either from the constants or from the cmd line args passed in eval 
    if test:
        if test_idxs is None: 
            test_idxs = constants.argsort_subject_ids[int(args['yaml_args']['data']['train_samples']):]
        else:
            test_idxs = [int(idx) for idx in test_idxs]
        test_dataset = itemgetter(*test_idxs)(dataset)
        if len(test_idxs) == 1:
            test_dataset = tuple([test_dataset])
        test_dataloader, n_win_sub_test = loader_from_dataset(
            args=args, 
            dataset=test_dataset, 
            valid_size=0, 
            shuffle=shuffle_test,
            torch_loader=True
        )
        assert len(n_win_sub_test) == len(test_idxs)
    else:
        test_dataloader, n_win_sub_test = None, None

    return train_dataloader, n_win_sub_train, test_dataloader, n_win_sub_test

