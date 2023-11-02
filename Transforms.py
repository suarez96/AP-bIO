from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from Signal import Signal
import neurokit2 as nk

class Transform(ABC):

    # Transforms such as subtracting mean and min-max scale should be here. As well as extract QRS-complex peaks and CWT
        
    def __init__(self):
        pass

    def __call__(self, signal):
        if isinstance(signal, Signal):
            if signal.transformed_data is None:
                signal.transformed_data = signal.data.copy()
            x = signal.transformed_data
        else:
            x = signal
            signal=None

        return self._transform(x, signal)
    
    def _transform(self, x):
        raise NotImplementedError
        

class Crop(Transform):
    
    def __init__(self, start, end=None, length=None, default_sample_rate=250):
        """
        start: start of crop in seconds
        end: end of crop in seconds
        length: length of crop in seconds
        """
        assert end or length, "One of 'end' or 'length' must be defined"
        self.start = start
        self.end = end
        self.length = length
        self.default_sample_rate = default_sample_rate

    def _transform(self, x, signal):
        try:
            sample_rate = signal.sample_rate
        except ValueError:#AttributeError:
            sample_rate = self.default_sample_rate
        
        if self.end is not None:
            return x[self.start*sample_rate:self.end*sample_rate]
        else:
            return x[self.start*sample_rate:self.start*sample_rate+self.length*sample_rate]

class SplineEnvelope(Transform):

    def __init__(self, n_spline_pts=None, peak_extraction_method=None):
        self.n_spline_pts = n_spline_pts
        self.peak_extraction_method = peak_extraction_method
    
    def extract_peaks(self, x):
        signal, info = nk.ecg_peaks(x, sampling_rate=250, correct_artifacts=True) 
        t = info['ECG_R_Peaks']
        # peaks and times
        return t, x[t]
        
    def _transform(self, x, signal):
        
        if self.n_spline_pts is None:
            self.n_spline_pts = x.shape[0]
            
        # peaks and times
        t, peaks = self.extract_peaks(x)
    
        # Creating a cubic spline interpolation
        cs = CubicSpline(t, peaks)
    
        # Generating new X values for the purpose of plotting a smooth spline
        t_new = np.linspace(t[0], t[-1], self.n_spline_pts)
    
        # Generating interpolated Y values
        spline = cs(t_new)
        return t_new, spline

class MinMaxScale(Transform):

    def __init__(self):
        pass
        
    def _transform(self, x, signal):
        return (x-x.min())/(x.max()-x.min())

class CWT(Transform):

    def __init__(self):
        pass

    def _transform(self, x, signal, plot=True):
        # TODO
        # make sure to remove the mean
        pass