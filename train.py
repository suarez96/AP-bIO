from Data import MarshData, FantasiaData
from Transforms import WPC
import Transforms
from Signal import Signal

import xgboost as xgb
import scipy
import matplotlib.pyplot as plt
import argparse
import os

# train_utils.py
import os
import numpy as np
from Signal import Signal
from numpy.lib.stride_tricks import as_strided
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
from tqdm.auto import tqdm

def eval(model, X_test, y_test, metrics={}):
    print("Training model...")
    y_pred = reg.predict(X_test)
    results = {}
    for metric_name, metric in metrics.items():
        results[metric_name] = metric(y_true, y_pred)
    print("Results")
    print(results)
    return results

def train(X_train, y_train, sample_size=50, hyper_params={}):
    print("Training model...")
    model = xgb.XGBRegressor(**hyper_params)
    model.fit(X_train[:sample_size], y_train[:sample_size])
    return model

def get_rolling_windows(
    signal: Signal, 
    window_size: int
):
        
    # Calculate new shape and strides
    new_shape = (signal.size - window_size + 1, window_size)
    new_strides = (signal.strides[0], signal.strides[0])
    
    # Create the rolling window view
    return as_strided(signal, shape=new_shape, strides=new_strides)

def process_ecg_and_ip(
    dataset, 
    idxs,
    ECG_transforms, 
    IP_transforms
):
    Xs, ys = [], []
    for train_idx in tqdm(idxs):
        
        # ECG input data
        input_ecg = dataset[train_idx].ECG().transform(transforms=ECG_transforms)
        # Breathing Rate input and target data
        input_ip = dataset[train_idx].IP().transform(transforms=IP_transforms)
        
        # rolling window parameters
        window_size = input_ecg.sample_rate*4
        
        Xs.append(
            get_rolling_windows(input_ecg.transformed_data, window_size=window_size)
        )
        ys.append(
            input_ip.transformed_data[window_size-1:]
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
    IP_transforms:list, 
    test_size: int=5, 
    random_seed: float=42
):
    # set random seed for reproducibility
    np.random.seed(seed=random_seed)
    all_idxs = np.random.choice(
        np.arange(len(dataset)), size=len(dataset), replace=False
    )
    train_idxs, test_idxs = all_idxs[:-test_size], all_idxs[-test_size:]
    
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
    
    return X_train, X_test, y_train, y_test

parser = argparse.ArgumentParser(description='Script to take test set samples from our DB and turn into json for multiple input.')
parser.add_argument('-s', '--start_time_crop', help='Start of crop for each sample, in seconds', default=240)
parser.add_argument('-e', '--end_time_crop', help='End of crop for each sample, in seconds', default=700)
parser.add_argument('-o', '--output_path', help='Output path', default=700)
args = parser.parse_args()

if __name__ == '__main__':

    transforms = [
        Transforms.Crop(args.start_time_crop, args.end_time_crop),
        Transforms.MinMaxScale(),
        lambda x: x-x.mean()
    ]

    # load all data
    marsh_dataset = [
        MarshData(f"../MARSH/{i}/") for i in os.listdir("../MARSH/") if len(i) == 4
    ]
    
    exclusion_list = [
        ('1436', 2),
        ('0046', 0),
        ('2655', 0),
        ('2655', 2),
        ('0048', 2),
        ('0037', 1),
        ('5329', 0)
    ]

    X_train, X_test, y_train, y_test = make_train_test_sets(
        dataset=marsh_dataset,
        ECG_transforms=transforms,
        IP_transforms=transforms, 
        test_size=5, 
        random_seed=42
    )

    model = train(X_train, y_train)
    results = eval(model, X_test, y_test,  metrics={"MSE": sklearn.metrics.mean_squared_error})
    export(model, input_shape=X_train.shape[1], path=args.output_path)