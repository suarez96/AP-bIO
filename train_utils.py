from Data import MarshData
import Transforms
from Signal import Signal
from typing import Union

import xgboost as xgb
import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
import onnxmltools
import onnxruntime as rt
from skl2onnx.common.data_types import FloatTensorType
from tqdm.auto import tqdm
import yaml

def load_yaml(path):

    with open(path) as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            return None
    
    return parsed_yaml

def predict(model_or_session, data):
    if isinstance(model_or_session, xgb.XGBRegressor):
        predictions = model_or_session.predict(data)
    elif isinstance(model_or_session, rt.InferenceSession):
        predictions = model_or_session.run(
            None, {'input': data.astype(np.float32)}
        )[0]
    else:
        raise RuntimeError(f"Unsupported model for inference: {type(model_or_session)}")
    return predictions


# @Deprecated
def evaluate(y_pred, y_test, metrics={}, results_file=None, params={}):
    run_id = hex(np.random.randint(0, 16**4))
    results = {}
    for metric_name, metric in metrics.items():
        results[metric_name] = metric(y_test, y_pred)
    if results_file is not None:
        df2 = {'run_id': run_id, **params, **results}
        df2 = pd.DataFrame.from_records([df2], index='run_id')
        if os.path.exists(results_file):
            df = pd.read_csv(results_file, index_col='run_id')
            combined_df = df.append(df2)
        else:
            combined_df = df2
        combined_df.to_csv(results_file)
        print(combined_df)
        print("Run Saved to file")
    print("Results")
    print(results)
    return results

# @Deprecated
def train(X_train, y_train, sample_size=None, hyper_params={}, randomize=True):
    print("Training model...")
    model = xgb.XGBRegressor(**hyper_params)
    # take random sample
    if sample_size is not None:
        if randomize:
            random_subset = np.random.choice(X_train.shape[0], sample_size, replace=False)
            X_subset, y_subset = X_train[random_subset], y_train[random_subset]
        else:
            X_subset, y_subset = X_train[:sample_size], y_train[:sample_size]
    else:
        X_subset, y_subset = X_train, y_train

    model.fit(X_subset, y_subset)
    return model


def get_rolling_windows(
        signal: Union[Signal, np.array],
        window_size: int,
        jump: int=1
):
    # Calculate new shape and strides
    new_shape = (signal.size - window_size + 1, window_size)
    
    # strides are the number of bytes needed to go from row n to row n+1, and to reach column n, to n+1
    new_strides = (signal.strides[0], signal.strides[0])

    # Create the rolling window view
    raw_strides = as_strided(signal, shape=new_shape, strides=new_strides)
    if jump > 1:
        return raw_strides[::jump]
        
    return raw_strides 


def process_ecg_and_ip(
        dataset,
        idxs,
        ECG_transforms,
        IP_transforms, 
        **kwargs
):
    Xs, ys = [], []
    for train_idx in tqdm(idxs):
        # ECG input data
        input_ecg = dataset[train_idx].ECG().transform(transforms=ECG_transforms)
        # Breathing Rate input and target data
        input_ip = dataset[train_idx].IP().transform(transforms=IP_transforms)

        # rolling window parameters
        window_size = input_ecg.sample_rate * 4

        Xs.append(
            get_rolling_windows(input_ecg.transformed_data, window_size=window_size)
        )
        ys.extend(
            input_ip.transformed_data[window_size - 1:]
        )
    return np.vstack(Xs), np.vstack(ys)


def export(model, input_shape, output_path):
    onnx_model = onnxmltools.convert_xgboost(
        model, initial_types=[('input', FloatTensorType([None, input_shape]))]
    )
    onnxmltools.utils.save_model(onnx_model, output_path)
    assert os.path.exists(output_path)
    print(f"Model saved successfully at: {output_path}")

    return None


def make_train_test_sets(
        dataset: list,
        ECG_transforms: list,
        IP_transforms: list,
        test_size: int = 5,
        random_seed: float = 42,
        randomize=True,
        train_idxs = []
):
    # set random seed for reproducibility
    np.random.seed(seed=random_seed)

    if randomize:
        all_idxs = np.random.choice(
            np.arange(len(dataset)), size=len(dataset), replace=False
        )
    else:
        all_idxs = np.arange(len(dataset))

    if train_idxs:
        test_idxs = [i for i in all_idxs if i not in train_idxs]
    else:
        train_idxs, test_idxs = all_idxs[:-test_size], all_idxs[-test_size:]
    print("Train indices:", train_idxs, "test indices:", test_idxs)
    X_train, y_train = process_ecg_and_ip(
        dataset,
        train_idxs,
        ECG_transforms,
        IP_transforms
    )

    X_test, y_test = process_ecg_and_ip(
        dataset,
        test_idxs,
        ECG_transforms,
        IP_transforms
    )

    return X_train, X_test, y_train, y_test, train_idxs, test_idxs


def build_model_name(args, test_idxs):
    """
    Construct model name based on the training input arguments
    :param args:
    :param test_idxs:
    :return:
    """
    if type(args) != dict:
        args = dict(args)
    name = args['output_path']
    for arg_name, arg_value in args.items():
        if arg_name != "output_path":
            name += f"_{arg_name}_{arg_value}"
    name += "_val_idxs_" + "_".join([str(i) for i in test_idxs])
    name += ".onnx"
    return name