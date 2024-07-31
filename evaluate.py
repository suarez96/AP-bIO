import numpy as np
import Transforms
from Signal import Signal
import matplotlib.pyplot as plt
import os
import logging
logger = logging.getLogger(__name__)

# TODO make each metric a callable
def evaluate_model(preds, gt, num_windows_per_subject=[], test_idxs=[], plot=False, save_visuals=False, model_name=None, **kwargs):
    start = 0
    scores = []
    if save_visuals:
        if not os.path.exists(f"visuals/{model_name}"):
            os.makedirs(f"visuals/{model_name}")

    for n_windows, test_idx in zip(num_windows_per_subject, test_idxs):
        end = start + n_windows
        # change from column to flat
        preds_subject, gt_subject = preds.flatten()[start:end], gt.flatten()[start:end]

        post_processing = [
            Transforms.ConvolveSmoothing(kernel_size=1000),
            Transforms.LowPass(cutoff=2),
            Transforms.Detrend(),
            Transforms.MinMaxScale(center=True),
        ]

        preds_subject = Signal(
            _type='IP', data=np.array(preds_subject), format='mat', filepath=None
        ).transform(transforms=post_processing)
        gt_subject = Signal(
            _type='IP', data=np.array(gt_subject), format='mat', filepath=None
        ).transform(transforms=post_processing)

        # center around 0 and plot
        # TODO make plotting separate function
        if plot or save_visuals:
            plt.figure(f"{model_name}_{test_idx}")
            plt.plot(preds_subject.transformed_data, label='Preds')
            plt.plot(gt_subject.transformed_data, label='GT')
            plt.title("Postprocessed Predictions vs Ground Truth")
            plt.legend()
            if save_visuals:
                plt.savefig(f"visuals/{model_name}/{test_idx}.png")
            if plot:
                plt.show()

        # TODO: fix cwt transform to not depend on signal sample_rate 
        preds_cwt, gt_cwt = Transforms.apply_cwt_transform(
            model_name, test_idx, preds_subject, gt_subject, plot=plot, save_visuals=save_visuals, **kwargs
        )


        score = Transforms.WPC(
            preds_cwt, gt_cwt, 
            freq=np.linspace(
                kwargs.get("low", 0.1), 
                kwargs.get("high", 0.55), 
                kwargs.get("resolution", 60)
            )
        )[2].mean()
        logger.info(f"Subject: {test_idx}, WPC: {score}")
        scores.append(score)
        # roll window to next sample
        start = end

    logger.info(f"Avg WPC: {np.array(scores).mean()}")
    return scores