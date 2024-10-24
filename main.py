import argparse
import train_utils
import Dataloader
import Models
import shutil # also move to export TODO
from evaluate import evaluate_model
import time
import logging
import copy
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='Script to train breathing signal predictor based on ECG input.'
)
# FIR high pass filter
# TODO add series processing to decimate resample => 250, 200, 150, 100, 50
# Highlight quantitative improvements (WPC, Instantaneuous frequency, MSE, max error)
# Step 1. Find top limit of SSA window size
# Step 2. Lowpass filter on input ECG to see if HF prediction artifacts disappear WITHOUT final postprocessing 
# Experiment A, no decimation. Experiment B, no decimation
parser.add_argument('-y', '--yaml_path', type=str, help='Filepath to YAML file with training run params', default='params.yml')
parser.add_argument('-e', '--eval-only', action='store_true', help='Only run evaluation, no training.')
parser.add_argument('-n', '--name', type=str, help='id/name of the model to be tested')
parser.add_argument('-f', '--framework', type=str, help='Model framework. Tsai, envelope, torch, etc.', default='tsai')
parser.add_argument('--train-indices', type=int, nargs="+", help='ID\'s of subjects to be trained on', default=None)
parser.add_argument('-i', '--test-indices', type=int, nargs="+", help='ID\'s of subjects to be tested', default=None)
parser.add_argument('-m', '--marsh_path', type=str, help='Filepath to MARSH root directory', default='../MARSH/')
parser.add_argument('-v', '--visualize', action='store_true', help='Plot helper visuals')
parser.add_argument('-s', '--save_visuals', action='store_true', help='Save eval visuals (Predictions and CWT)')

args = vars(parser.parse_args())

def run(args):

    start_time = time.time()
    if args['eval_only']:
        args['yaml_args'] = train_utils.load_yaml(
            f"params/{args['name']}_params.yml"
        )
    else:
        args['yaml_args'] = train_utils.load_yaml(
            args['yaml_path']
        )

    builder_args = dict(
        marsh_path=args['marsh_path'],
        train_samples=args['yaml_args']['data']['train_samples'],
        seq_len=args['yaml_args']['model_params']['seq_len'],
        global_ecg_pipeline=args['yaml_args']['global_ecg_pipeline'],
        global_ip_pipeline=args['yaml_args']['global_ip_pipeline'],
        framework=args['framework'],
        jump_size=args['yaml_args']['learner_params']['jump_size'],
        batch_size=args['yaml_args']['learner_params']['batch_size'],
        visualize=args['visualize']
    )

    train_loader_builder = Dataloader.LoaderBuilder(eval_mode=False, **builder_args)
    test_loader_builder = Dataloader.LoaderBuilder(eval_mode=True, **builder_args)
    # train logic
    if not args['eval_only']:

        # TODO make train/test loaders separately since they each use different jump sizes. Could remove train/test boolean args # TODO add data leakage check
        # build train dataloader
        train_loader, _, _ = train_loader_builder.build_loaders(
            train=True,
            idxs=args['train_indices']
        )

        # untrained model
        if args['framework'] == 'tsai':
            model = Models.TSAITransformer(
                dataloader=train_loader,
                model_params=args['yaml_args']['model_params'],
                framework=args['framework']
            )
        elif args['framework'] == 'envelope':
            model = Models.ECGEnvelopeModel(framework=args['framework'])
        else:
            raise AssertionError("Training halted. Framework must be one of tsai/envelope.")

        shutil.copyfile(args['yaml_path'], f'params/{model.run_id}_params.yml')
        print(f"Opening log for model id: {model.run_id}")
        logging.basicConfig(filename=f'logs/{model.run_id}_train.log', level=logging.INFO)
        if args['train_indices'] is not None:
            logger.info(f"Train indices: {str(list(args['train_indices']))}")
        # train logic inside
        model.train(
            args['yaml_args']['learner_params']['iters'], 
            args['yaml_args']['learner_params']['lr']
        )
        # save model
        model.export()
        framework = model.framework
        model_name = model.run_id
    else:
        model = None
        framework = args['framework']
        model_name = args['name']
        logging.basicConfig(filename=f'logs/{model_name}_eval.log', level=logging.INFO)
    
    # test logic
    # build test loader
    test_loader, test_num_windows_per_subject, adjusted_indices = test_loader_builder.build_loaders(
        train=False,
        valid_size=0,
        shuffle=False,
        torch_loader=True,
        idxs=args['test_indices']
    )
    # load pretrained model

    if args['framework'] == 'tsai':
        eval_model = Models.TSAITransformer(
            path=f"models/{framework}/{model_name}.pkl", 
            cpu=False,
            model_params=args['yaml_args']['model_params']
        )
    elif args['framework'] == 'envelope':
        eval_model = Models.ECGEnvelopeModel(
            framework=args['framework'],
            path=f"models/{framework}/{model_name}.pkl", 
            model_params=args['yaml_args']['model_params']
        )
    # create predictions
    preds_full, gt_full = eval_model.infer(dataloader=test_loader)
    # calculate metrics. ATM only WPC is calculated
    scores = evaluate_model(
        preds=preds_full, 
        gt=gt_full,
        num_windows_per_subject=test_num_windows_per_subject,
        test_idxs=adjusted_indices,
        plot=args['visualize'],
        save_visuals=args['save_visuals'],
        model_name=model_name, 
        experiment_description=args['yaml_args'].get('desc', ''),
        post_processing_pipeline=args['yaml_args']['post_processing_pipeline'],
        **args['yaml_args']['cwt_evaluation'], 
    )
    print(f"DONE! \n{scores}")

    end_time = time.time()  
    elapsed_time = end_time - start_time 
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    
    run(args)