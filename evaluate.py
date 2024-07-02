import numpy as np
import Transforms
from Signal import Signal
import matplotlib.pyplot as plt
import os
from functools import partial
import logging
logger = logging.getLogger(__name__)

def create_cwt_transform(model_name, test_idx, data_type, plot=False, save_visuals=False, **kwargs):
    return Transforms.CWT(
        plot=plot,
        save_visuals=save_visuals,
        lower_bound=kwargs.get("low", 0.1),
        higher_bound=kwargs.get("high", 0.55),
        resolution=kwargs.get("resolution", 60),
        model_name=model_name,
        test_idx=test_idx,
        data_type=data_type
    )

# TODO make each metric a callable
<<<<<<< HEAD
def evaluate_model(preds, gt, device='cpu', num_windows_per_subject=[], test_idxs=[], plot=False, save_visuals=False, model_name=None, post_processing_pipeline=[], **kwargs):
=======
def evaluate_model(preds, gt, num_windows_per_subject=[], test_idxs=[], plot=False, save_visuals=False, model_name=None, **kwargs):
>>>>>>> 5c38695 (save_visuals flag to save eval visuals)
    start = 0
    scores = []
    if save_visuals:
        if not os.path.exists(f"visuals/{model_name}"):
            os.makedirs(f"visuals/{model_name}")

    for n_windows, test_idx in zip(num_windows_per_subject, test_idxs):
        end = start + n_windows
        # change from column to flat
        preds_subject, gt_subject = preds.flatten()[start:end], gt.flatten()[start:end]

        post_processing = Transforms.build_transforms(
            post_processing_pipeline
        )

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
<<<<<<< HEAD
            plt.plot(preds_subject.transformed_data, label='predictions smoothed')
            plt.plot(gt_subject.transformed_data, label='ground truth')
=======
            plt.plot(preds_subject.transformed_data, label='Preds')
            plt.plot(gt_subject.transformed_data, label='GT')
>>>>>>> 5c38695 (save_visuals flag to save eval visuals)
            plt.title("Postprocessed Predictions vs Ground Truth")
            plt.legend()
            if save_visuals:
                plt.savefig(f"visuals/{model_name}/{test_idx}.png")
            if plot:
                plt.show()

        # TODO: fix cwt transform to not depend on signal sample_rate 
<<<<<<< HEAD
        cwt = Transforms.CWT(
            device=device,
            plot=plot, 
            lower_bound=kwargs.get("low", 0.1), 
            higher_bound=kwargs.get("high", 0.55), 
            resolution=kwargs.get("resolution", 60)
=======
        create_cwt_transform_partial = partial(
            create_cwt_transform,
            model_name=model_name,
            test_idx=test_idx,
            plot=plot,
            save_visuals=save_visuals,
            **kwargs
>>>>>>> 5c38695 (save_visuals flag to save eval visuals)
        )

        preds_cwt = create_cwt_transform_partial(data_type='Preds')(preds_subject)
        gt_cwt = create_cwt_transform_partial(data_type='GT')(gt_subject)

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