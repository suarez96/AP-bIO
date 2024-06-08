from tsai.all import MSELossFlat, rmse, TST, Learner, ShowGraph, load_learner
from abc import ABC, abstractmethod
import numpy as np
import os
import logging
import Transforms
from Signal import Signal
import matplotlib.pyplot as plt

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
        # the fastai object that manages the training loop
        if self.path is None:
            self.model = TST(dataloader.vars, dataloader.c, seq_len=seq_len)
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

    # TODO change to "infer" move eval logic to separate file
    def eval(self, dataloader, num_windows_per_subject=[], test_idxs=[], **kwargs):
        logger.info("Evaluating model")
        preds_full, gt_full = self.learner.get_preds(dl=dataloader)
        start = 0
        scores = []
        for n_windows, test_idx in zip(num_windows_per_subject, test_idxs):
            end = start + n_windows
            # change from column to flat
            preds, gt = preds_full.flatten()[start:end], gt_full.flatten()[start:end]

            post_processing = [
                Transforms.ConvolveSmoothing(kernel_size=500),
                Transforms.MinMaxScale(center=True), 
            ]

            preds = Signal(
                _type='IP', data=np.array(preds), format='mat', filepath=None
            ).transform(transforms=post_processing)
            gt = Signal(
                _type='IP', data=np.array(gt), format='mat', filepath=None
            ).transform(transforms=post_processing)

            # center around 0 and plot
            # TODO make plotting separate function
            # plt.plot(preds.transformed_data, label='predictions')
            # plt.plot(gt.transformed_data, label='ground truth')
            # plt.title("After scaling")
            # plt.legend()
            # plt.show()
            # TODO: fix cwt transform to not depend on signal sample_rate 
            cwt = Transforms.CWT(
                plot=False, 
                lower_bound=kwargs.get("low", 0.1), 
                higher_bound=kwargs.get("high", 0.55), 
                resolution=kwargs.get("resolution", 60)
            )
            preds_cwt = cwt(preds)
            gt_cwt = cwt(gt)
            score = Transforms.WPC(preds_cwt, gt_cwt)[2].mean()
            logger.info(f"Subject: {test_idx}, WPC: {score}")
            scores.append(score)
            # roll window to next sample
            start = end
        return scores

    def export(self):
        model_path = os.path.join(self.export_dir_root, self.framework, f"{self.run_id}.pkl")
        if os.path.exists(model_path):
            model_path.replace(".pkl", "_new.pkl")
        logger.info(f"{model_path} saved")
        self.learner.export(model_path)
        

class BidirectionalRNN(Model):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
