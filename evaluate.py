import numpy as np
import Transforms
from Signal import Signal
import matplotlib.pyplot as plt
import os
import logging
import pandas as pd
from datetime import datetime


logger = logging.getLogger(__name__)

# TODO make each metric a callable
def evaluate_model(preds, gt, experiment_description='', device='cpu', output_file='compiled_results.csv', num_windows_per_subject=[], test_idxs=[], plot=False, save_visuals=False, model_name=None, post_processing_pipeline=[], **kwargs):
    start = 0
    wpc_scores = []
    mse_scores = []
    if save_visuals:
        if not os.path.exists(f"visuals/{model_name}"):
            os.makedirs(f"visuals/{model_name}")

    if not os.path.exists(output_file):
        results_df = pd.DataFrame()
    else:
        results_df = pd.read_csv(output_file, index_col='Model ID')
    # new results row for this run
    row = {"Model ID": model_name, "Description": experiment_description, "Train Date": datetime.today().strftime('%Y-%m-%d')}

    for n_windows, test_idx in zip(num_windows_per_subject, test_idxs):
        end = start + n_windows
        # change from column to flat
        preds_subject, gt_subject = preds.flatten()[start:end], gt.flatten()[start:end]

        post_processing = Transforms.build_transforms(
            pipeline=post_processing_pipeline, eval_mode=True
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
            plt.plot(preds_subject.transformed_data, label='predictions smoothed')
            plt.plot(gt_subject.transformed_data, label='ground truth')
            plt.title(f"Subject {test_idx}: Postprocessed Predictions vs Ground Truth - {model_name}")
            plt.legend()
            if save_visuals:
                plt.savefig(f"visuals/{model_name}/{test_idx}.png")
            if plot:
                plt.show()
            plt.clf()
        # TODO: fix cwt transform to not depend on signal sample_rate 
        def apply_cwt(data, data_type):
            return Transforms.CWT(
                device=device,
                plot=plot, 
                lower_bound=kwargs.get("low", 0.1), 
                higher_bound=kwargs.get("high", 0.55), 
                resolution=kwargs.get("resolution", 60),
                model_name=model_name,
                test_idx=test_idx,
                save_visuals=save_visuals,
                data_type=data_type
            )(data)

        preds_cwt = apply_cwt(preds_subject, "Predictions")
        gt_cwt = apply_cwt(gt_subject, "Ground Truth")

        mse = Transforms.MeanSquaredError(
            gt_subject.transformed_data, preds_subject.transformed_data
        )
        logger.info(f"Subject: {test_idx}, MSE: {mse}")
        mse_scores.append(mse)
        score = Transforms.WPC(
            preds_cwt, gt_cwt, 
            freq=np.linspace(
                kwargs.get("low", 0.1), 
                kwargs.get("high", 0.55), 
                kwargs.get("resolution", 60)
            )
        )[2].mean()
        logger.info(f"Subject: {test_idx}, WPC: {score}")
        wpc_scores.append(score)

        # add results to dataframe row
        row[f"MSE {test_idx}"] = mse
        row[f"WPC {test_idx}"] = score

        # roll window to next sample
        start = end

    # calculate aggregate scores
    avg_wpc = np.array(wpc_scores).mean()
    avg_mse = np.array(mse_scores).mean()
    logger.info(f"Avg WPC: {avg_wpc}")
    logger.info(f"Avg MSE: {avg_mse}")

    # save aggregate results to df
    row["AVG WPC"] = avg_wpc
    row["AVG MSE"] = avg_mse
    row_dataframe = pd.DataFrame.from_records([row], index="Model ID")
    results_df = pd.concat([results_df, row_dataframe])
    results_df.to_csv(output_file)

    return f"WPC Scores: {wpc_scores}\nAvg WPC: {np.array(wpc_scores).mean()} \nMSE Scores: {mse_scores} \nAvg MSE: {avg_mse}"