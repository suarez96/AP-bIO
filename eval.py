import argparse
from train import run
import train_utils
import Dataloader
import Models
import shutil # also move to export TODO

import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='Script to train breathing signal predictor based on ECG input.'
)

parser.add_argument('-n', '--name', type=str, help='id/name of the model to be tested')
parser.add_argument('-i', '--indices', type=int, nargs="+", help='ID\'s of subjects to be tested', default=[1436, 5111, 256, 8722])
parser.add_argument('-m', '--marsh_path', type=str, help='Filepath to MARSH root directory', default='../MARSH/')
parser.add_argument('-v', '--visualize', action='store_true', help='Plot helper visuals')
args = vars(parser.parse_args())

def run(args):

    args['yaml_args'] = train_utils.load_yaml(
        f"params/{args['name']}_params.yml", eval=True
    )
    _, _, test_loader, num_windows_per_subject = Dataloader.build_loaders(args, train=False, test=True, test_idxs=args['indices'])
    print("num_windows_per_subject", num_windows_per_subject)
    model = Models.TSAITransformer(
        dataloader=test_loader, 
        seq_len=args['yaml_args']['hparams']['seq_len'], 
        path=f"models/tsai/{args['name']}.pkl", 
        cpu=False
    )
    print(f"Log: {model.run_id}")
    logging.basicConfig(filename=f'logs/{model.run_id}_eval.log', level=logging.INFO)
    # assert False
    scores = model.eval(
        test_loader, 
        **args['yaml_args']['cwt_evaluation'], 
        num_windows_per_subject=num_windows_per_subject,
        test_idxs=args['indices'],
        plot=args['visualize']
    )
    print(f"DONE! \nScores: {scores}")


if __name__ == '__main__':
    
    run(args)