from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from Signal import Signal
import neurokit2 as nk
import pywt
import matplotlib.pyplot as plt

class Transform(ABC):
    """
    Base class. Abstract
    """
        
    def __init__(self):
        pass

    def __call__(self, signal):
        """
        Is simply an interface for the transform function. Before _transform is called, __call__ checks if the input is a signal object 
        or a raw signal.
        """
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
    """
    Based on charlton work, the spline envelope based on the ECG signal is used as a reference for the breathing rate.
    """

    def __init__(self, n_spline_pts=None, **kwargs):
        self.n_spline_pts = n_spline_pts
        self.peak_extraction_method = kwargs.get("peak_extraction_method", "martinez2004")
        self.correct_artifacts = kwargs.get("correct_artifacts", False)
    
    def extract_peaks(self, x, sample_rate=250):
        """
        https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/ecg/ecg_peaks.html#ecg_peaks
        """
        signal, info = nk.ecg_peaks(
            x, 
            sampling_rate=sample_rate, 
            correct_artifacts=self.correct_artifacts, 
            method=self.peak_extraction_method
        )
        t = info['ECG_R_Peaks']
        # peaks and times
        return  np.hstack([np.zeros(1), t]), np.hstack([np.ones(1)*x.mean(), x[t]])
        
    def _transform(self, x, signal):
        
        if self.n_spline_pts is None:
            self.n_spline_pts = x.shape[0]
            
        # peaks and times
        t, peaks = self.extract_peaks(x, sample_rate=signal.sample_rate)
    
        # Creating a cubic spline interpolation
        cs = CubicSpline(t, peaks)
    
        # Generating new X values for the purpose of plotting a smooth spline
        t_new = np.linspace(t[0], t[-1], self.n_spline_pts)
    
        # Generating interpolated Y values
        spline = cs(t_new)
        return t_new, spline

class MeanSubtraction(Transform):
    """
    In practice, removes DC. This method will make any signal zero-mean.
    """

    def __init__(self):
        pass
        
    def _transform(self, x, signal):
        return x-x.mean()

class MinMaxScale(Transform):
    """
    Scales any signal from 0 to 1, not incredibly useful in practice, but very useful for visualizations.
    """

    def __init__(self):
        pass
        
    def _transform(self, x, signal):
        return (x-x.min())/(x.max()-x.min())


class CWT(Transform):
    """
    Calculates the continuous wavelet transform based on the parameters provided in the constructor. Default wavelet is morlet with A=0.5, B=0.8125.
    """

    def __init__(
        self, 
        lower_bound: float=0.1, 
        higher_bound: float=1, 
        resolution: int=10, 
        plot: bool=False, 
        wavelet: str='cmor5-0.8125'
    ):
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound
        self.resolution = resolution
        self.plot = plot
        self.wavelet = wavelet
        wavelet_params = self.wavelet.split('cmor')[1]
        self.wavelet_A_param = float(wavelet_params.split('-')[0])
        self.wavelet_B_param = float(wavelet_params.split('-')[1])

    def _transform(self, x, signal):
        # remove DC level
        x -= x.mean()
        freq_space = np.linspace(self.lower_bound, self.higher_bound, self.resolution)
        scales = (self.wavelet_B_param*signal.sample_rate)/freq_space
        coefficients, frequencies = pywt.cwt(x, scales, self.wavelet, sampling_period=1/signal.sample_rate)
        coefficients = np.abs(coefficients)[::-1, :] # flip to get ascending frequency
        if self.plot:
            plt.figure(figsize=(9, 3))
            plt.imshow(
                coefficients,  # Replace with your data or coefficients
                aspect='auto',
                extent=[0, coefficients.shape[1]/signal.sample_rate, self.lower_bound, self.higher_bound], 
                interpolation='bilinear',
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Freq (hz)")
            plt.title(f'CWT {signal.type}')
            plt.show()
        return coefficients