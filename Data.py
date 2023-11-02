from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
import plotly.graph_objects as go
import os
from Signal import Signal
import Transforms

class Data(ABC):
    
    # interface for signals
    def __init__(self, root):
        self.data = {
            "ECG": None,
            'ECG_annot': None,
            'ECG_ENV': None, # envelope
            "PPG": None,
            "IP": None,
            'IP_annot': None,
            "NASAL": None,
            'NASAL_annot': None,
        }
        self.root = root

    def ECG(self):
        return self.data['ECG']

    def PPG(self):
        return self.data['PPG']

    def IP(self):
        return self.data['IP']

    def NASAL(self):
        return self.data['NASAL']

    def ECG_ENV(self):
        return self.data['ECG_ENV']

    # should have an option according to which type of signal we want to plot
    def plot_all(self, time_ranges=[None], sample_rate=250):
        raise NotImplementedError


class MarshData(Data):

    def __init__(self, root, **kwargs):
        """
        While waiting for MIMIC 3

        Source
        https://www.sciencedirect.com/science/article/pii/S1746809420300434
        Section 3.2.1
        
        This dataset, 'MARSH', includes respiratory signals as part of the study "Fusion enhancement for tracking of 
        respiratory rate through intrinsic mode functions in photoplethysmography." It is meant to support academic research, 
        particularly on algorithm development tools.
        
        Contents:
        
        Data.txt (age, gender, height, weight, systole, diastole, [respectively])
        ECG.mat (raw ECG data)
        ECG_annot.mat (annotations for the R peaks in ECG data)
        IP.mat (Raw IP data)
        IP_annot.mat (annotations for the local maxima of IP data [end of inspiration phase])
        NASAL.mat (Thermistor mask data)
        NASAL_annot.mat (annotations for the local maxima of thermistor mask data [end of inspiration phase])
        PPG.mat (Raw PPG signal data)
        
        Sample_rates ECG, IP is 250hz
        
        PPG is 500hz
        """
        super().__init__(root)
        # This is the file structure of each sample in MARSH
        self.sample_rate_map = {
            'ECG': 250,
            'ECG_annot': 0,
            'IP': 250,
            'IP_annot': 0,
            'NASAL': 500,
            'NASAL_annot': 0,
            'PPG': 500
        }
        # load each signal using the load mat function
        for fn, sample_rate in self.sample_rate_map.items():
            meta = fn.split('.')[0]
            self.data[meta] = Signal(
                format='mat', filepath=os.path.join(self.root, fn+'.mat'), type=meta, sample_rate=sample_rate
            )

        _, ecg_envelope = Transforms.SplineEnvelope(**kwargs)(self.ECG().data)
        self.data['ECG_ENV'] = Signal(
            data=ecg_envelope, format='mat', filepath="N/A", type='ECG_ENV', sample_rate=self.sample_rate_map['ECG']
        )


class FantasiaData(Data):

    def __init__(self):
        # TODO    
        super().__init__()
    
        self.song_limits = [
            0,
            9*60 + 20,
            15*60 + 30,
            8*60 + 30,
            23*60 + 15,
            3*60,
            8*60 + 30,
            9*60 + 30,
            11*60 + 30,
        ]
        
        self.song_names = [
            "Toccata and Fugue in D Minor",
            "The Nutcracker Suite",
            "The Sorcerer's Apprentice",
            "Rite of Spring",
            "Intermission/Meet the Soundtrack",
            "The Pastoral Symphony",
            "Dance of the Hours",
            "Night on Bald Mountain and Ave Maria"
        ]

    def load(self):
        pass
        # TODO
        # for fp in tqdm(
        # glob.glob("C:/Users/augus/Google Drive/UofT/MEng/fantasia/physionet.org/files/fantasia/1.0.0/*dat")[:3]
        # ):
        #     fp_trimmed = fp.split('.dat')[0]
        #     fant_data, info = wfdb.rdsamp(os.path.abspath(fp_trimmed))
        #     fant_br, fant_ecg = fant_data.T
        #     fant_br = fant_br[:fant_ecg.shape[0]-fant_ecg.shape[0]%Fs_fantasia]
        #     fant_ecg = fant_ecg[:fant_ecg.shape[0]-fant_ecg.shape[0]%Fs_fantasia]
            
        #     _, envelope = spline_envelope(fant_ecg, n_spline_pts=fant_ecg.shape[0])


class Dataset:

    # to organize samples from a single data source,
    # getitem should call the load function from each of the data classes
    # 
    def __init__(self):
        # TODO
        pass
        # will go through a root directory, loading all samples

    def __getitem__(self, i):
        # TODO
        pass
