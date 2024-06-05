from tsai.all import MSELossFlat, rmse, TST, Learner, ShowGraph
from abc import ABC, abstractmethod
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)

class Model(ABC):

    def __init__(self, export_dir_root='models', **kwargs):
        self.run_id = hex(np.random.randint(0, 8**8))
        self.framework = "no_framework"
        self.export_dir_root = export_dir_root

    def eval():
        raise NotImplementedError

    def train():
        raise NotImplementedError

    def export(self):
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
        self.learner = Learner(dataloader, self.model, loss_func=MSELossFlat(), metrics=rmse)

    def train(self, iters, lr):
        logger.info("Training finished")
        self.learner.fit_one_cycle(iters, lr)
        logger.info("Training finished")

    def export(self):
        model_path = os.path.join(self.export_dir_root, self.framework, f"{self.run_id}.pkl")
        if os.path.exists(model_path):
            model_path.replace(".pkl", "_new.pkl")
        logger.info(f"{model_path} saved")
        self.learner.export(model_path)

class BidirectionalRNN(Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
