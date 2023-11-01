from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from Signal import Signal

class Transform(ABC):

    # Transforms such as subtracting mean and min-max scale should be here. As well as extract QRS-complex peaks and CWT
        
    def __init__(self):
        pass

    def __call__(self, signal):
        if isinstance(signal, Signal):
            if signal.transformed_data is None:
                signal.transformed_data = signal.data.copy()
            x = signal.transformed_data
        return self._transform(x)
    
    def _transform(self, x):
        raise NotImplementedError
        

class Crop(Transform):
    
    def __init__(self, start, end=None, length=None):
        """
        start: start of crop in seconds
        end: end of crop in seconds
        length: length of crop in seconds
        """
        assert end or length, "One of 'end' or 'length' must be defined"
        self.start = start
        self.end = end
        self.length = length

    def _transform(self, signal):
        if self.end is not None:
            return x[self.start*signal.sample_rate:self.end*signal.sample_rate]
        else:
            return x[self.start*signal.sample_rate:self.start*signal.sample_rate+self.length*signal.sample_rate]

class SplineEnvelope(Transform):

    def __init__(self, n_spline_pts=None, peak_extraction_method=None):
        self.n_spline_pts = n_spline_pts
        self.peak_extraction_method = peak_extraction_method
    
    def extract_peaks(self, x):
        # TODO
        
        # peaks and times
        return t, peaks
        
    def _transform(self, x):
        
        if n_spline_pts is None:
            n_spline_pts = x.shape[0]
            
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
        
    def _transform(self, x):
        return (x-x.min())/(x.max()-x.min())

class CWT(Transform):

    def __init__(self):
        pass

    def _transform(self, x, plot=True):
        # TODO
        # make sure to remove the mean
        pass