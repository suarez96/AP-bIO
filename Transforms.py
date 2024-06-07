from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import CubicSpline
from Signal import Signal
import neurokit2 as nk
import pywt
import matplotlib.pyplot as plt
import math
import scipy
from tqdm import tqdm

def build_transforms(pipeline=None, pipeline_args=None, search_space=None):
    """
    Use the parameters defined in our yaml or in our search space to build
    the transforms
    """

    assert pipeline_args is None or search_space is None, "Conflicting Pipeline Hyperparameters. "

    # used for matching string arguments with the python
    translation_map = {
        'Crop': Crop,
        'MeanSubtraction': MeanSubtraction,
        'MinMaxScale': MinMaxScale,
        'Detrend': Detrend
    }

    created_pipeline = []
    for transform_name, transform_params in pipeline.items():

        # unpack nonetype or list of dicts into a single dict
        args = {}
        if transform_params is not None:     
            for p in transform_params:
                args.update(p)

        created_pipeline.append(
            translation_map[transform_name](**args)
        )

    return created_pipeline

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
        super().__init__()
        self.start = start
        self.end = end
        self.length = length
        self.default_sample_rate = default_sample_rate

    def _transform(self, x, signal):
        try:
            sample_rate = signal.sample_rate
        except AttributeError:
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
        super().__init__()
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
        super().__init__()
        
    def _transform(self, x, signal):
        return x-x.mean(axis=-1, keepdims=True)

class Detrend(Transform):
    """
    In practice, removes DC. This method will make any signal zero-mean.
    """

    def __init__(self):
        super().__init__()
        self.fn = scipy.signal.detrend
        
    def _transform(self, x, signal):
        return self.fn(x)

class MinMaxScale(Transform):
    """
    Scales any signal from 0 to 1, not incredibly useful in practice, but very useful for visualizations.
    """

    def __init__(self):
        super().__init__()
        
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
        wavelet: str='cmor5-0.8125',
        sample_rate=None,
        **kwargs
    ):
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound
        self.resolution = resolution
        self.plot = plot
        self.wavelet = wavelet
        wavelet_params = self.wavelet.split('cmor')[1]
        self.wavelet_A_param = float(wavelet_params.split('-')[0])
        self.wavelet_B_param = float(wavelet_params.split('-')[1])
        self.freq_space = np.linspace(self.lower_bound, self.higher_bound, self.resolution)
        if sample_rate is None:
            self.scales = None
        else:
            self.scales = (self.wavelet_B_param*sample_rate)/self.freq_space

    def _transform(self, x, signal):
        # remove DC level
        x -= x.mean()
        if self.scales is None:
            self.scales = (self.wavelet_B_param*signal.sample_rate)/self.freq_space
        coefficients, frequencies = pywt.cwt(x, self.scales, self.wavelet, sampling_period=1/signal.sample_rate)
        if self.plot:
            plt.figure(figsize=(9, 3))
            plt.imshow(
                np.abs(coefficients)[::-1, :],  # flip axis
                aspect='auto',
                extent=[0, coefficients.shape[1]/signal.sample_rate, self.lower_bound, self.higher_bound], 
                interpolation='bilinear',
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Freq (hz)")
            plt.title(f'CWT {signal.type}')
            plt.show()
        return coefficients

# static transforms
def phase_coherence(angles):
    return np.sqrt((np.sum(np.cos(angles)))**2 + (np.sum(np.sin(angles)))**2) / np.shape(angles)[0]

def angles_between_wavelet_coefficients(coeffs1, coeffs2):
    return np.angle(coeffs1) - np.angle(coeffs2)

def WPC_of_coeffs(coeffs1, coeffs2):
    return phase_coherence(angles_between_wavelet_coefficients(coeffs1, coeffs2))


def WPC(cwt1, cwt2, fs=250, freq=np.linspace(0.1, 0.55, 60), num_cyc=5):
   
    coeffs1 = cwt1
    coeffs2 = cwt2
   
    [num_freqs, sig_len] = coeffs1.shape
    window_size = (1./freq)*num_cyc
   
    # number of samples for the PC matrix based on smallest window
    x0 = num_cyc * fs / freq[-1]
    M = math.floor(sig_len / x0)
    PC = np.zeros([num_freqs, M])
    PC_norm = np.zeros([num_freqs, M])
    t = np.linspace(0, M/freq[-1], PC_norm.shape[1])
    
    for f_idx in tqdm(range(freq.shape[0]), desc="Calculating WPC"):
        # split sample into windows of num_cyc cycles
        win_len_sec = window_size[f_idx] # in seconds
        win_len = math.floor(win_len_sec * fs) # in samples
        num_windows = math.floor(sig_len / win_len)
        pc_f = np.zeros([num_windows])
        for w_idx in range(num_windows):
            start_idx = w_idx*win_len
            end_idx = start_idx + win_len
            #print(start_idx, end_idx)
            pc_f[w_idx] = WPC_of_coeffs(coeffs1[f_idx,start_idx:end_idx],
                                        coeffs2[f_idx,start_idx:end_idx])
       
        resampled_pc_f = np.clip(scipy.signal.resample_poly(pc_f, M, num_windows), 0, 1)
       
        # consider whether any magnitudes > 1
        PC[f_idx,:] = np.transpose(resampled_pc_f)

    return t, None, PC

def gauss_kernel(l=5, sig=1):
    """
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def upsample_and_blur(t=None, a=None, x_rep=3, y_rep=3, kernel_size=5):
    if a is not None:
        upsample_a = np.repeat(a, y_rep, axis=1).repeat(x_rep, axis=0)
    else:
        upsample_a = None

    if t is not None:
        upsample_t = np.linspace(t[0], t[-1], upsample_a.shape[1])
    else:
        upsample_t = None
    if upsample_a is not None:
        upsample_a = scipy.signal.convolve2d(
            upsample_a,
            gauss_kernel(l=kernel_size),
            mode='same'
        )
        
    return upsample_t, upsample_a