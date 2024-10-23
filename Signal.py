import numpy as np
from scipy.io import loadmat
import plotly.graph_objects as go
import os
from types import LambdaType
import plotly.express as px
import wfdb
import copy

class Signal:
    """
    Signal object. A combination of raw bioelectric signals, with its metadata such as source, filetype, sample rate, the type of 
    recording (ECG, PPG). Also includes some useful plotting functions such as plot (time domain plot) and fft (freq domain plot).
    """

    color_map = {
            "ECG": px.colors.qualitative.Plotly[0],
            'ECG_ENV': px.colors.qualitative.Plotly[3], # envelope
            "PPG": px.colors.qualitative.Plotly[1],
            "IP": px.colors.qualitative.Plotly[2],
            "NASAL": px.colors.qualitative.Plotly[4],
        }
    
    
    def __init__(
        self, 
        format: str,
        filepath: str, 
        _type: str, 
        data: np.array = None,
        sample_rate: int = 250,
        verbose=True
    ):
        self.format = format
        self.filepath = filepath
        self.type = _type
        self.sample_rate = sample_rate
        self.verbose = verbose
        try:
            self.load_fn = {
                'mat': self.mat_loader,
                'dat': self.dat_loader
            }[self.format]
        except KeyError:
            raise KeyError("Format must be one of: "+ ','.join(list(self.load_fn.keys())))
        self.data = data if data is not None else self.load_fn(self.filepath, _type)
        self.transformed_data=None
        self.is_original = True
    
    def plot(self, start_time = 0, base_fig=None, transformed=False):
        """
        start_time: start time in seconds
        base_fig: figure to add traces to
        """
        if transformed:
            assert self.transformed_data is not None, "Data must be transformed before calling 'plot' with transformed=True"
        
        x = self.transformed_data if transformed else self.data
        t = np.arange(start_time, start_time+x.shape[0]/self.sample_rate, 1/self.sample_rate)
        if base_fig is None:
            base_fig = go.Figure(
                layout=dict(
                    title=f'Signals: {self.filepath}',
                    xaxis_title='Time (s)',
                    yaxis_title='Signal Value'
                )
            )
            base_fig.update_layout(plot_bgcolor='rgba(240, 240, 240, 0.8)', showlegend=True)
            base_fig.update_xaxes(showline=True, mirror=True, linewidth=1, linecolor='black')
            base_fig.update_yaxes(showline=True, mirror=True, linewidth=1, linecolor='black')
        
        base_fig.add_trace(
            go.Scatter(
                x=t,
                y=x,
                name=self.type,
                marker=dict(color=self.color_map[self.type])
            )
        )
        base_fig.update_traces(opacity=0.8)
        return base_fig

    def __repr__(self):
        return str({
            "filepath": self.filepath,
            "data": self.data,
            "type": self.type,
            "sample_rate": self.sample_rate,
        })

    def copy(self):
        return copy.copy(self)

    def transform(self, transforms, inplace=False, visualize=False):
        """
        Apply a series of transforms defined in the pipeline called "transforms".
        """
        # create single plot for all step by step visualizations
        if visualize:
            all_transforms_fig = go.Figure(layout=dict(title='All transforms'))
            all_transforms_fig.add_trace(
                go.Scatter(
                    y = self.data.copy(),
                    name = "Initial Input Data"
                )
            )
        else:
            all_transforms_fig = None

        self.transformed_data = self.data.copy()
        for t in transforms:
            # lambda function support
            if isinstance(t, LambdaType):
                self.transformed_data = t(self.transformed_data)
            else:
                self.transformed_data = t(self)
            if visualize and all_transforms_fig is not None:
                # plot current transforms
                all_transforms_fig.add_trace(
                    go.Scatter(
                        y = self.transformed_data,
                        name = str(t)
                    )
                )

        # replace data with transformed data
        if inplace:
            self.data=self.transformed_data

        if visualize:
            all_transforms_fig.show()
        return self

    def mat_loader(self, filepath, _type):
        """
        Load from .mat file
        """
        x = loadmat(filepath)[_type]
        if self.verbose: print(self.type, "shape: ", x.shape)
        return x.flatten()
        
    def dat_loader(self, filepath, _type):
        """
        filepath: str requires a file with a .dat and a .head part
        """
        return wfdb.rdsamp(os.path.abspath(filepath))

    def fft(self, transformed=False, top_freq=5):
        """
        plot the fast fourier transform for the signal data
        """
        x = (self.transformed_data if transformed else self.data).copy()
        x -= x.mean()
        M = abs(np.fft.fft(x))
        freq = np.fft.fftfreq(x.shape[0], d=1/self.sample_rate)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x = freq[(freq > 0) & (freq < top_freq)],
                y = M[(freq > 0) & (freq < top_freq)]
            )
        )
        fig.show()