import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
import plotly.graph_objects as go
import os

class Signal:
    def __init__(
        self, 
        format: str,
        filepath: str, 
        type: str, 
        data: np.array = None,
        sample_rate: int = 250
    ):
        self.format = format
        self.filepath = filepath
        self.type = type
        self.sample_rate = sample_rate
        try:
            self.load_fn = {
                'mat': self.mat_loader,
                'dat': self.dat_loader
            }[self.format]
        except KeyError:
            raise KeyError("Format must be one of: "+ ','.join(list(self.load_fn.keys())))
        self.data = data if data is not None else self.load_fn(self.filepath, type)
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
        
        base_fig.add_trace(
            go.Scatter(
                x=t,
                y=x,
                name=self.type
            )
        )
        base_fig.update_layout(showlegend=True)
        return base_fig

    def __repr__(self):
        return str({
            "filepath": self.filepath,
            "data": self.data,
            "type": self.type,
            "sample_rate": self.sample_rate,
        })

    def transform(self, transforms, inplace=False):
        self.transformed_data = self.data.copy()
        for t in transforms:
            self.transformed_data = t(self)
        if inplace:
            self.data=self.transformed_data
        return self

    def mat_loader(self, filepath, type):
        x = loadmat(filepath)[type]
        print(self.type, "shape: ", x.shape)
        return x.flatten()
        
    def dat_loader(self, filepath):
        return wfdb.rdsamp(os.path.abspath(fp_trimmed))