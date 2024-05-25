from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
import plotly.graph_objects as go
import os
from Signal import Signal
import Transforms

class Data(ABC):
    """
    Basically an interface for signal objects.
    """
    
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

    def ECG_annot(self):
        return self.data['ECG_annot']

    def IP_annot(self):
        return self.data['IP_annot']

    def NASAL_annot(self):
        return self.data['NASAL_annot']

    # should have an option according to which type of signal we want to plot
    def plot_all(self, time_ranges=[None], sample_rate=250):
        raise NotImplementedError


class MarshData(Data):
    """
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

    def __init__(self, root, **kwargs):
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
                format='mat',
                filepath=os.path.join(self.root, fn+'.mat'),
                _type=meta,
                sample_rate=sample_rate,
                verbose=kwargs.get("verbose", True)
            )

        _, ecg_envelope = Transforms.SplineEnvelope(**kwargs)(self.ECG())
        self.data['ECG_ENV'] = Signal(
            data=ecg_envelope,
            format='mat',
            filepath="N/A",
            _type='ECG_ENV',
            sample_rate=self.sample_rate_map['ECG'],
            verbose=kwargs.get("verbose", True)
        )


class FantasiaData(Data):
    """
    Fantasia Database expanded (March 2, 2003, midnight)

    A subset of data from the Fantasia Database has been available here for several years; the remainder of the database 
    is now available. It consists of ECG and respiration recordings, with beat annotations, from 20 young and 20 elderly 
    subjects, all rigorously screened as healthy, in sinus rhythm during a resting state (two hours each). Half of the 
    recordings also include (uncalibrated) continuous noninvasive blood pressure signals.

    From  Virtual Respiratory Rate Sensors 5-A: An Example of A Smartphone-Based Integrated and 
    Multiparametric mHealth Gateway
    
    A. Analysis on the PhysioNet Fantasia Database
        The PhysioNet Fantasia database contains data of
        young (21–34 year old) and elderly (68–85) healthy subjects,
        who underwent 120 min of continuous supine rest, while
        the ECG signal and respiration signals were recorded with
        clinical instrumentation at a sample rate of 250 Hz. The
        respiratory activity has been acquired as well, by using a
        respiration belt [18]. 
    """
    
    songs = {
        1: {"name": "Toccata and Fugue in D Minor", "times": {"start": 0, "end": 9*60 + 20}},
        2: {"name": "The Nutcracker Suite", "times": {"start": 9*60 + 20, "end": 15*60 + 30}},
        3: {"name": "The Sorcerer's Apprentice", "times": {"start": 15*60 + 30, "end": 8*60 + 30}},
        4: {"name": "Rite of Spring", "times": {"start": 8*60 + 30, "end": 23*60 + 15}},
        5: {"name": "Intermission/Meet the Soundtrack", "times": {"start": 23*60 + 15, "end": 3*60}},
        6: {"name": "The Pastoral Symphony", "times": {"start": 3*60, "end": 8*60 + 30}},
        7: {"name": "Dance of the Hours", "times": {"start": 8*60 + 30, "end": 9*60 + 30}},
        8: {"name": "Night on Bald Mountain and Ave Maria", "times": {"start": 9*60 + 30, "end": 11*60 + 30}}
    }

    def __init__(self, root, **kwargs):
        super().__init__(root)
        
        self.sample_rate_map = {
            'ECG': 250,
            'IP': 250
        }        
        
        # load each signal using wfdb.rdsamp(os.path.abspath(fp_trimmed)) function
        # fantasia only has 2 signal types, both at 250hz
        joined_signal = Signal(
            format='dat',
            filepath=self.root,
            _type='N/A',
            sample_rate=250,
            verbose=kwargs.get("verbose", True)
        )
        ecg_signal = joined_signal.copy()
        ip_signal = joined_signal.copy()
        ip_signal.data, ecg_signal.data = joined_signal.data[0].T
        self.patient_dict = joined_signal.data[1]
        ecg_signal.type, ip_signal.type = 'ECG', 'IP'

        self.data['ECG'] = ecg_signal
        self.data['IP'] = ip_signal
        _, ecg_envelope = Transforms.SplineEnvelope(**kwargs)(self.ECG())
        self.data['ECG_ENV'] = Signal(
            data=ecg_envelope,
            format='mat',
            filepath="N/A",
            _type='ECG_ENV',
            sample_rate=self.sample_rate_map['ECG'],
            verbose=kwargs.get("verbose", True)
        )


class Dataset:
    """
    To organize samples from a single data source.
    __getitem__ should call the load function from each of the data classes
    """

    # 
    def __init__(self):
        # TODO
        pass
        # will go through a root directory, loading all samples

    def __getitem__(self, i):
        # TODO
        pass
