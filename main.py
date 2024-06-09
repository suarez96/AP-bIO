import argparse
from train import run
import train_utils
import Dataloader
import Models
import shutil # also move to export TODO
from evaluate import evaluate_model

import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='Script to train breathing signal predictor based on ECG input.'
)

parser.add_argument('-y', '--yaml_path', type=str, help='Filepath to YAML file with training run params', default='params.yml')
parser.add_argument('-e', '--eval-only', action='store_true', help='Only run evaluation, no training.')
parser.add_argument('-n', '--name', type=str, help='id/name of the model to be tested')
parser.add_argument('-f', '--framework', type=str, help='Model framework. Tsai, torch, etc.', default='tsai')
parser.add_argument('-i', '--indices', type=int, nargs="+", help='ID\'s of subjects to be tested', default=[1436, 5111, 256, 8722])
parser.add_argument('-m', '--marsh_path', type=str, help='Filepath to MARSH root directory', default='../MARSH/')
parser.add_argument('-v', '--visualize', action='store_true', help='Plot helper visuals')

args = vars(parser.parse_args())

def run(args):

    # train logic
    if not args['eval_only']:

        args['yaml_args'] = train_utils.load_yaml(
            args['yaml_path']
        )
        # TODO make train/test loaders separately since they each use different jump sizes. Could remove train/test boolean args
        # TODO add data leakage check
        # build train dataloader
        train_loader, _, _, _ = Dataloader.build_loaders(
            args,
            train=not args['eval_only'],
            test=True,
            test_idxs=args['indices']
        )
        # blank model
        model = Models.TSAITransformer(
            dataloader=train_loader, 
            seq_len=args['yaml_args']['hparams']['seq_len']
        )
        # TODO MOVE TO EXPORT METHOD
        shutil.copyfile(args['yaml_path'], f'params/{model.run_id}_params.yml')
        print(f"Opening log for model id: {model.run_id}")
        logging.basicConfig(filename=f'logs/{model.run_id}_train.log', level=logging.INFO)
        # train logic inside
        model.train(args['yaml_args']['hparams']['iters'], args['yaml_args']['hparams']['lr'])
        # save model
        model.export()
        framework = model.framework
        model_name = model.run_id
    else:
        model = None
        framework = args['framework']
        model_name = args['name']
    
    # test logic
    args['yaml_args'] = train_utils.load_yaml(
        f"params/{model_name}_params.yml", eval=True
    )
    # build test loader
    _, _, test_loader, test_num_windows_per_subject = Dataloader.build_loaders(
        args,
        train=not args['eval_only'],
        test=True,
        test_idxs=args['indices']
    )
    # load pretrained model
    eval_model = Models.TSAITransformer(
        seq_len=args['yaml_args']['hparams']['seq_len'], 
        path=f"models/{framework}/{model_name}.pkl", 
        cpu=False
    )
    logging.basicConfig(filename=f'logs/{eval_model.run_id}_eval.log', level=logging.INFO)
    # create predictions
    preds_full, gt_full = eval_model.infer(dataloader=test_loader)
    # calculate metrics. ATM only WPC is calculated
    scores = evaluate_model(
        preds=preds_full, 
        gt=gt_full,
        num_windows_per_subject=test_num_windows_per_subject,
        test_idxs=args['indices'],
        plot=args['visualize'],
        **args['yaml_args']['cwt_evaluation'], 
    )
    print(f"DONE! \nScores: {scores}")


if __name__ == '__main__':
    
    run(args)