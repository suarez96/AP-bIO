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

parser.add_argument('-y', '--yaml_path', type=str, help='Filepath to YAML file with training run params', default='params.yml')
parser.add_argument('-m', '--marsh_path', type=str, help='Filepath to MARSH root directory', default='../MARSH/')
args = vars(parser.parse_args())

def run(args):

    args['yaml_args'] = train_utils.load_yaml(args['yaml_path'])
    train_loader, _, test_loader, _ = Dataloader.build_loaders(args)
    model = Models.TSAITransformer(
        dataloader=train_loader, 
        seq_len=args['yaml_args']['hparams']['seq_len']
    )
    # TODO MOVE TO EXPORT METHOD
    shutil.copyfile(args['yaml_path'], f'params/{model.run_id}_params.yml')
    print(f"Log: {model.run_id}")
    logging.basicConfig(filename=f'logs/{model.run_id}_train.log', level=logging.INFO)
    model.train(args['yaml_args']['hparams']['iters'], args['yaml_args']['hparams']['lr'])
    # assert False
    # model.eval(test_loader, args['metrics'])
    model.export()



if __name__ == '__main__':
    
    run(args)