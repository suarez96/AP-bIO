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
parser.add_argument('-i', '--indices', type=int, nargs="+", help='ID\'s of subjects to be tested')
parser.add_argument('-m', '--marsh_path', type=str, help='Filepath to MARSH root directory', default='../MARSH/')
args = vars(parser.parse_args())

def run(args):

    args['yaml_args'] = train_utils.load_yaml(f"params/{args['name']}_params.yml")
    _, test_loader = Dataloader.build_loaders(args, train=False, test=True, test_idxs=args['indices'])
    model = Models.TSAITransformer(dataloader=test_loader, seq_len=args['yaml_args']['hparams']['seq_len'], path=f"models/tsai/{args['name']}.pkl")
    print(f"Log: {model.run_id}")
    logging.basicConfig(filename=f'logs/{model.run_id}_eval.log', level=logging.INFO)
    # assert False
    # model.eval(test_loader, args['metrics'])



if __name__ == '__main__':
    
    run(args)