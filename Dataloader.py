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
import matplotlib.pyplot as plt


class LoaderBuilder:

    def __init__(
        self,
        marsh_path,
        train_samples,
        seq_len,
        global_ecg_pipeline,
        global_ip_pipeline,
        jump_size,
        batch_size,
        framework,
        visualize=False
    ):
        self.marsh_path = marsh_path
        self.train_samples = train_samples
        self.full_dataset = {
            int(i): MarshData(os.path.join(marsh_path, i), verbose=False) for i in tqdm(os.listdir(self.marsh_path), desc='traversing MARSH data...') if len(i) == 4
        }
        self.global_ecg_pipeline = Transforms.build_transforms(
            global_ecg_pipeline
        )
        self.global_ip_pipeline = Transforms.build_transforms(
            global_ip_pipeline
        )
        print("self.global_ecg_pipeline", self.global_ecg_pipeline)
        print("self.global_ip_pipeline", self.global_ip_pipeline)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.jump_size = jump_size
        self.framework = framework
        self.visualize=visualize


    def build_ECG_input_windows(
            self,
            dataset,
            jump_size
        ):
        """
        Build the rolling windows from the ECG data inside each of the subject
        signals.
        Dataset: a list of MarshData or FantasiaData objects to iterate through
        """

        # matrices to feed to the dataloader
        X_stack = []
        y_stack = []
        num_windows_per_subject = []

        # take global ECG and break it into parts after applying transform
        for subject in tqdm(dataset, desc="building dataloader..."):
            
            subject_id = int(subject.ECG().filepath.split('/')[-2])

            # build targets
            input_ip_raw = subject.IP().transform(transforms=self.global_ip_pipeline)
            input_ip = input_ip_raw.transformed_data
            y_ip = input_ip[self.seq_len-1:][::jump_size]
            y_stack.append(y_ip)

            # build inputs. will be the ECG envelope if that model is selected
            if self.framework == 'envelope':
                input_envelope_raw = subject.ECG_ENV().transform(transforms=self.global_ip_pipeline)
                input_envelope = input_envelope_raw.transformed_data[self.seq_len-1:][::jump_size]
                X_stack.append(input_envelope)
                num_windows_per_subject.append(len(input_envelope))

            else:
                input_ecg_raw = subject.ECG().transform(transforms=self.global_ecg_pipeline)
                # extract transformed data
                input_ecg = input_ecg_raw.transformed_data
                # get rolling windows
                X_ecg_rolling = get_rolling_windows(
                    input_ecg, 
                    window_size=self.seq_len, 
                    jump=jump_size
                )
                X_stack.append(X_ecg_rolling)
                num_windows_per_subject.append(len(X_ecg_rolling))

            if self.visualize:
                print('SHAPES', X_stack[-1][-1].shape, y_ip.shape)
                # # plt.plot(X_stack[-1], label='Postprocessed Input')

                fig, ax = plt.subplots(nrows=2, figsize=(9, 6))
                fig.suptitle(f"Postprocessed {subject_id}")
                ax[0].plot(X_stack[-1][0])
                ax[0].plot(X_stack[-1][len(X_stack)//2])
                ax[0].plot(X_stack[-1][-1])
                ax[0].set_title(f'Postprocessed Input Samples')
                ax[1].plot(y_ip)
                ax[1].set_title(f'Postprocessed Target')
                fig.tight_layout()
                fig.show()


        return np.vstack(X_stack), np.stack(y_stack).flatten(), num_windows_per_subject


    def loader_from_dataset(
        self,
        dataset,
        valid_size,
        jump_size,
        tfms=[None, TSRegression()],
        batch_tfms=TSStandardize(by_sample=True), # Standardize data TODO add more batch transforms
        shuffle=True,
        torch_loader=False,
        **kwargs,
    ):

        """
        Construct the dataloader based on a train or test list of MarshData/FantasiaData objects.
        shuffle: shuffle data in the loader
        """

        X_ecg_rolling_stack_np, y_ip_stack_np, num_windows_per_subject = self.build_ECG_input_windows(dataset=dataset, jump_size=jump_size)
        X_array = X_ecg_rolling_stack_np.reshape(-1, 1, 1 if self.framework=='envelope' else self.seq_len)
        y_array = y_ip_stack_np.reshape(-1, 1)
        assert X_array.shape[0] == y_array.shape[0], "Inputs and Target shapes do not match!"
        
        # use pytorch dataloader
        # for evaluation. Avoids the strange index error 
        if torch_loader:
            X_tensor = torch.tensor(X_array).float()  # Convert to float32 tensor
            y_tensor = torch.tensor(y_array).float()  # Convert to float32 tensor
            dls = DataLoader(
                TensorDataset(X_tensor, y_tensor), 
                batch_size=self.batch_size
            )
        # use fastai type dataloader
        else:
            splits = get_splits(
                y_array,
                valid_size=valid_size, 
                stratify=True, 
                random_state=23, 
                shuffle=shuffle, 
                show_plot=False
            )
            # TODO investigate this step
            
            # create TSDatasets from X, y then put into a TSDataloader
            dls = get_ts_dls(
                X_array, 
                y_array, 
                splits=None if not valid_size else splits, 
                tfms=tfms, 
                batch_tfms=batch_tfms, 
                bs=self.batch_size
            )
        return dls, num_windows_per_subject

    def build_loaders(self, valid_size=0.2, train: bool=True, shuffle: bool=True, idxs: list=None, torch_loader: bool=False, **kwargs):

        """
        test: whether to build a test dataloader or return None
        shuffle_test (bool): whether to shuffle the data in the test loader. usuallly false because we want to build the predictions
        in time order
        """

        if idxs is None:
            if train:
                idxs = constants.argsort_subject_ids_train[:int(self.train_samples)]
            else:
                idxs = constants.argsort_subject_ids_train[int(self.train_samples):] + constants.argsort_subject_ids_test

        else:
            # cast all idxs to int in case
            idxs = [int(idx) for idx in idxs]

        dataset = itemgetter(*idxs)(self.full_dataset)
        if len(idxs) == 1:
            dataset = tuple([dataset])

        dataloader, n_windows_per_subject = self.loader_from_dataset(
            dataset=dataset,
            valid_size=valid_size, 
            shuffle=shuffle,
            torch_loader=torch_loader,
            jump_size=self.jump_size if train else 1,
            **kwargs
        )
        assert len(n_windows_per_subject) == len(idxs)

        return dataloader, n_windows_per_subject



