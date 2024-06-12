from tsai.all import MSELossFlat, rmse, TST, Learner, ShowGraph, load_learner
from abc import ABC, abstractmethod
import numpy as np
import os
import logging
import Transforms
from Signal import Signal
import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)

# callback for logging
from fastai.callback.core import Callback
class LogEpochMetrics(Callback):
    def after_epoch(self):
        logger.info(",".join(self.recorder.metric_names))
        logger.info(self.recorder.log)

class Model(ABC):

    def __init__(self, export_dir_root='models', path=None, **kwargs):
        self.path = path
        if path is not None:
            self.run_id = path.split("/")[-1].replace(".pkl", "")
        else:
            self.run_id = hex(np.random.randint(0, 8**8))
        self.framework = "no_framework"
        self.export_dir_root = export_dir_root

    def infer():
        raise NotImplementedError

    def train():
        raise NotImplementedError

    def export(self):
        # TODO add params.yml saving as well
        raise NotImplementedError

class TSAITransformer(Model):

    def __init__(self, dataloader=None, model_params={}, **kwargs):
        """
        export_dir_root (str): the parent directory where models will be saved
        """
        super().__init__(**kwargs)
        self.framework = 'tsai'
        # the fastai object that manages the training loop
        if self.path is None:
            self.model = TST(dataloader.vars, dataloader.c, **model_params)
            self.learner = Learner(
                dataloader,
                self.model, 
                loss_func=MSELossFlat(), 
                metrics=rmse,
                cbs=LogEpochMetrics()
            )
        else:
            self.learner = load_learner(self.path, cpu=kwargs.get("cpu", True))
            self.learner.model.eval()

    def train(self, iters, lr):
        logger.info("Training finished")
        self.learner.fit_one_cycle(iters, lr)
        self.learner.model.eval()
        logger.info("Training finished")

    def infer(self, dataloader, num_windows_per_subject=[], test_idxs=[], plot=False,**kwargs):
        logger.info("Evaluating model")
        with torch.no_grad():
            preds_full, gt_full = self.learner.get_preds(dl=dataloader)
        return preds_full, gt_full

    def export(self):
        model_path = os.path.join(self.export_dir_root, self.framework, f"{self.run_id}.pkl")
        if os.path.exists(model_path):
            model_path.replace(".pkl", "_new.pkl")
        logger.info(f"{model_path} saved")
        self.learner.export(model_path)
        

class BidirectionalRNN(Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ECGEnvelopeModel(Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.framework='envelope'

    def train(self):
        pass

    def export(self):
        pass

    def infer(self, dataloader, num_windows_per_subject=[], test_idxs=[], plot=False,**kwargs):
        logger.info("Evaluating model")
        # return ECG envelope attribute if data is Signal, else calculate ECG ENV and return it for each sample