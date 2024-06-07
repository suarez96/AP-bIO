from tsai.all import MSELossFlat, rmse, TST, Learner, ShowGraph, load_learner
from abc import ABC, abstractmethod
import numpy as np
import os
import logging
import Transforms
from Signal import Signal

logger = logging.getLogger(__name__)

class Model(ABC):

    def __init__(self, export_dir_root='models', path=None, **kwargs):
        self.path = path
        if path is not None:
            self.run_id = path.split("/")[-1].replace(".pkl", "")
        else:
            self.run_id = hex(np.random.randint(0, 8**8))
        self.framework = "no_framework"
        self.export_dir_root = export_dir_root

    def eval():
        raise NotImplementedError

    def train():
        raise NotImplementedError

    def export(self):
        # TODO add params.yml saving as well
        raise NotImplementedError

class TSAITransformer(Model):

    def __init__(self, dataloader, seq_len=256, **kwargs):
        """
        export_dir_root (str): the parent directory where models will be saved
        """
        super().__init__(**kwargs)
        self.framework = 'tsai'
        self.model = TST(dataloader.vars, dataloader.c, seq_len=seq_len)
        # the fastai object that manages the training loop
        if self.path is None:
            self.learner = Learner(
                dataloader,
                self.model, 
                loss_func=MSELossFlat(), 
                metrics=rmse
            )
        else:
            self.learner = load_learner(self.path, cpu=kwargs.get("cpu", True))

    def train(self, iters, lr):
        logger.info("Training finished")
        self.learner.fit_one_cycle(iters, lr)
        logger.info("Training finished")

    def eval(self, dataloader, **kwargs):
        logger.info("Evaluating model")
        preds, gt = self.learner.get_preds(dl=dataloader)
        cwt = Transforms.CWT(plot=True, lower_bound=kwargs.get("lower_bound", 0.1), higher_bound=kwargs.get("higher_bound", 0.55), resolution=kwargs.get("resolution", 60))
        # remove crop on preds and gt
        preds = Signal(_type='IP', data=np.array(preds), format='mat', filepath=None) # TODO: fix cwt transform to not depend on signal sample_rate 
        preds_cwt = cwt(preds)
        gt = Signal(_type='IP', data=np.array(gt), format='mat', filepath=None)
        gt_cwt = cwt(gt)
        print(preds_cwt.shape, gt_cwt.shape)
        return
        # return Transforms.WPC(preds_cwt, gt_cwt) TODO: Debug WPC

    def export(self):
        model_path = os.path.join(self.export_dir_root, self.framework, f"{self.run_id}.pkl")
        if os.path.exists(model_path):
            model_path.replace(".pkl", "_new.pkl")
        logger.info(f"{model_path} saved")
        self.learner.export(model_path)
        

class BidirectionalRNN(Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
