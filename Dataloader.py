from Data import MarshData, FantasiaData
from Transforms import WPC
import Transforms
from Signal import Signal
from train_utils
from tsai.all import get_splits, TSRegression, TSStandardize, get_ts_dls


def build_windows(
        dataset: list
    ):
    """
    Build the rolling windows from the ECG data inside each of the subject
    signals.
    Dataset: a list of MarshData or FantasiaData objects to iterate through

    """

    # matrices to feed to the dataloader
    X_ecg_rolling_train_stack = []
    y_ip_train_stack = []

    # take global ECG and break it into parts after applying transform
    for subject in dataset:

        input_ecg_raw = subject.ECG().transform(transforms=[
            # should take transforms from args, i.e. from params.yml
            Transforms.Crop(start_time_XGB, end_time_XGB), # C
            Transforms.MinMaxScale(), # S
            lambda x: x-x.mean(), # M
            lambda x: detrend(x), # D
        ])

        input_ip_raw = subject.IP().transform(transforms=[
            Transforms.Crop(start_time_XGB, end_time_XGB),
            Transforms.MinMaxScale(),
            lambda x: x-x.mean(),
            lambda x: detrend(x),
        ])

        # extract transformed data
        input_ecg = input_ecg_raw.transformed_data
        input_ip = input_ip_raw.transformed_data

        # get rolling windows
        X_ecg_rolling = get_rolling_windows(
            input_ecg, 
            window_size=args['hparams']['seq_len'], 
            jump=args['hparams']['train_jump_size']
        )
        y_ip = input_ip.data[window_size-1:][::train_jump_size]

        X_ecg_rolling_train_stack.append(X_ecg_rolling)
        y_ip_train_stack.append(y_ip)

    return np.vstack(X_ecg_rolling_train_stack), np.stack(y_ip_train_stack).flatten()

def build_loader(args):

    # build_loader ()
    dataset = [
        MarshData(f"../MARSH/{i}/") for i in os.listdir("../MARSH/") if len(i) == 4
    ][:4] # TODO: REMOVE INDEXING WHEN DONE

    X_ecg_rolling_train_stack_np, y_ip_train_stack_np = build_windows(dataset)
    tsai_X_train = X_ecg_rolling_train_stack_np.reshape(-1, 1, args['hparams']['seq_len'])
    tsai_y_train = y_ip_train_stack_np.reshape(-1, 1)
    splits = get_splits(tsai_y_train, valid_size=0.2, stratify=True, random_state=23, shuffle=True)
    tfms  = [None, TSRegression()]
    # TODO add more batch transforms
    batch_tfms = TSStandardize(by_sample=True)  # Standardize data
    dls = get_ts_dls(tsai_X_train, tsai_y_train, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=batch_size)
    return dls